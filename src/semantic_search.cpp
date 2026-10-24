#include "semantic_search.h"

#include "paths.h"
#include "sci_c.h"

#include "ExecutionProviders.h"
#include "ModelZoo.h"
#include "SemanticHeatmapColors.h"
#include "SentenceSplitter.h"

#include <gdk/gdkkeysyms.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kIndicator = 20;
constexpr long kMaxBytes = 2L * 1024L * 1024L;
constexpr size_t kMaxSentences = 4096;
constexpr unsigned kQueryDebounceMs = 300;
constexpr unsigned kRebuildDebounceMs = 600;
constexpr size_t kCacheLimit = 12288;

struct Span {
    int64_t byte_start = 0;
    int64_t byte_length = 0;
};

struct MainCall {
    std::function<void()> fn;
};

gboolean dispatch_main(gpointer data)
{
    auto *call = static_cast<MainCall *>(data);
    call->fn();
    delete call;
    return G_SOURCE_REMOVE;
}

void post_main(std::function<void()> fn)
{
    g_idle_add(dispatch_main, new MainCall{std::move(fn)});
}

class Worker {
public:
    Worker() : thread_([this] { loop(); }) {}
    ~Worker()
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }
    void post(std::function<void()> fn)
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back(std::move(fn));
        }
        cv_.notify_one();
    }

private:
    void loop()
    {
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                fn = std::move(queue_.front());
                queue_.pop_front();
            }
            fn();
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    bool stop_ = false;
    std::thread thread_;
};

struct Controller {
    Worker worker;

    GtkWidget *revealer = nullptr;
    GtkWidget *entry = nullptr;
    GtkWidget *status = nullptr;
    GtkWidget *spinner = nullptr;
    GtkWidget *model_dd = nullptr;
    GtkWidget *sens_dd = nullptr;
    GtkWidget *ep_dd = nullptr;
    GtkWidget *swatch[6] = {};
    GtkCssProvider *legend_css = nullptr;

    bool ui_ready = false;
    bool open = false;
    int sensitivity = 0;
    unsigned band_mask = 0;
    std::string model_id;
    std::string ep_id = "auto";
    std::vector<std::string> model_ids;
    std::vector<std::string> ep_ids;

    GtkWidget *sci = nullptr;
    NppDoc *doc = nullptr;
    std::vector<Span> spans;
    int64_t first_id = 1;
    int64_t next_id = 1;
    bool index_ready = false;
    std::string query;
    std::string index_status;
    std::vector<SemanticHit> last_hits;
    std::map<NppDoc *, std::set<int>> our_bookmarks;

    std::atomic<uint64_t> build_gen{0};
    std::atomic<uint64_t> query_gen{0};
    guint query_timer = 0;
    guint rebuild_timer = 0;

    ModelZoo zoo;
    std::unique_ptr<EmbeddingProvider> provider;
    std::unique_ptr<SimilarityEngine> engine;
    std::unique_ptr<VectorIndex> index;
    std::unordered_map<std::string, std::vector<float>> cache;
    std::string loaded_model;
    std::string loaded_ep;
    std::string backend_name = "ONNX Runtime CPU";

    sptr_t send(GtkWidget *w, unsigned msg, uptr_t wp, sptr_t lp)
    {
        return scintilla_send_message(SCINTILLA(w), msg, wp, lp);
    }

    void set_status(const std::string &text, bool busy)
    {
        if (status) {
            gtk_label_set_text(GTK_LABEL(status), text.c_str());
            gtk_widget_set_tooltip_text(status, text.c_str());
        }
        if (spinner) {
            gtk_widget_set_visible(spinner, busy);
            if (busy) gtk_spinner_start(GTK_SPINNER(spinner));
            else gtk_spinner_stop(GTK_SPINNER(spinner));
        }
    }

    void save_prefs()
    {
        gchar *path = npp_local_file(nullptr, "semantic-search.ini");
        GKeyFile *kf = g_key_file_new();
        g_key_file_set_integer(kf, "semantic", "sensitivity", sensitivity);
        g_key_file_set_string(kf, "semantic", "model", model_id.c_str());
        g_key_file_set_string(kf, "semantic", "provider", ep_id.c_str());
        g_key_file_save_to_file(kf, path, nullptr);
        g_key_file_free(kf);
        g_free(path);
    }

    void load_prefs()
    {
        gchar *path = npp_local_file(nullptr, "semantic-search.ini");
        GKeyFile *kf = g_key_file_new();
        if (g_key_file_load_from_file(kf, path, G_KEY_FILE_NONE, nullptr)) {
            GError *err = nullptr;
            int sens = g_key_file_get_integer(kf, "semantic", "sensitivity", &err);
            if (!err) sensitivity = std::clamp(sens, -1, 1);
            g_clear_error(&err);
            gchar *model = g_key_file_get_string(kf, "semantic", "model", nullptr);
            gchar *ep = g_key_file_get_string(kf, "semantic", "provider", nullptr);
            if (model && *model) model_id = model;
            if (ep && *ep) ep_id = ep;
            g_free(model);
            g_free(ep);
        }
        g_key_file_free(kf);
        g_free(path);
    }

    void configure_indicator()
    {
        if (!sci) return;
        /* Full-box indicators go through Surface::AlphaRectangle. The GTK4
         * Scintilla surface still stubs that call, so a wash would not
         * paint. Text foreground uses the same per-sentence colour and
         * does draw. */
        send(sci, SCI_INDICSETSTYLE, kIndicator, INDIC_TEXTFORE);
        send(sci, SCI_INDICSETFLAGS, kIndicator, SC_INDICFLAG_VALUEFORE);
    }

    void sync_bookmarks(const std::set<int> &want)
    {
        if (!sci || !doc) return;
        const int marker = SC_MARKNUM_BOOKMARK;
        const unsigned mask = 1u << marker;
        std::set<int> &ours = our_bookmarks[doc];
        for (auto it = ours.begin(); it != ours.end();) {
            if (!want.count(*it)) {
                send(sci, SCI_MARKERDELETE, static_cast<uptr_t>(*it), marker);
                it = ours.erase(it);
            } else {
                ++it;
            }
        }
        for (int line : want) {
            if (ours.count(line)) continue;
            if (static_cast<unsigned>(send(sci, SCI_MARKERGET, static_cast<uptr_t>(line), 0)) & mask)
                continue;
            send(sci, SCI_MARKERADD, static_cast<uptr_t>(line), marker);
            ours.insert(line);
        }
    }

    SemanticHeatmap::Profile heatmap_profile() const
    {
        const ModelSpec *spec = zoo.find(model_id);
        return spec ? spec->heatmap_profile : SemanticHeatmap::Profile::MiniLM;
    }

    void refresh_legend()
    {
        if (!legend_css) return;
        GString *css = g_string_new("");
        for (int i = 0; i < 6; ++i) {
            double sample = SemanticHeatmap::legendScore(heatmap_profile(), i);
            uint32_t bgr = SemanticHeatmap::colorBGR(sample, 0, heatmap_profile());
            g_string_append_printf(css,
                "button.sem-band-%d { background: #%02x%02x%02x;"
                " min-width: 16px; min-height: 16px; padding: 0; border-radius: 3px; }"
                "button.sem-band-%d:checked { box-shadow: inset 0 0 0 2px currentColor; }",
                i, bgr & 255, (bgr >> 8) & 255, (bgr >> 16) & 255, i);
        }
        gtk_css_provider_load_from_string(legend_css, css->str);
        g_string_free(css, TRUE);
    }

    void paint(const std::vector<SemanticHit> &hits)
    {
        if (!sci) return;
        sptr_t doc_len = send(sci, SCI_GETLENGTH, 0, 0);
        send(sci, SCI_SETINDICATORCURRENT, kIndicator, 0);
        send(sci, SCI_INDICATORCLEARRANGE, 0, doc_len);
        std::set<int> want_lines;
        for (const SemanticHit &hit : hits) {
            int band = SemanticHeatmap::bandForScore(hit.score, heatmap_profile());
            if (band_mask && (band < 0 || !(band_mask & (1u << band)))) continue;
            if (hit.sentence_id < first_id) continue;
            size_t idx = static_cast<size_t>(hit.sentence_id - first_id);
            if (idx >= spans.size()) continue;
            const Span &span = spans[idx];
            if (span.byte_start >= doc_len || span.byte_length <= 0) continue;
            long len = std::min(span.byte_length, static_cast<int64_t>(doc_len - span.byte_start));
            uint32_t color = SemanticHeatmap::colorBGR(hit.score, sensitivity, heatmap_profile());
            send(sci, SCI_SETINDICATORVALUE,
                 static_cast<uptr_t>(color | SC_INDICVALUEBIT), 0);
            send(sci, SCI_INDICATORFILLRANGE, static_cast<uptr_t>(span.byte_start), len);
            if (band_mask) {
                int line = static_cast<int>(send(sci, SCI_LINEFROMPOSITION,
                                                 static_cast<uptr_t>(span.byte_start), 0));
                want_lines.insert(line);
            }
        }
        sync_bookmarks(want_lines);
    }

    void clear_heatmap()
    {
        last_hits.clear();
        if (!sci) return;
        send(sci, SCI_SETINDICATORCURRENT, kIndicator, 0);
        send(sci, SCI_INDICATORCLEARRANGE, 0, send(sci, SCI_GETLENGTH, 0, 0));
        sync_bookmarks({});
    }

    std::string storage_dir(const std::string &id) const
    {
        gchar *base = npp_local_dir();
        gchar *path = g_build_filename(base, "semantic-models", id.c_str(), nullptr);
        std::string out = path;
        g_free(path);
        g_free(base);
        return out;
    }

    void schedule_query()
    {
        if (query_timer) g_source_remove(query_timer);
        const char *text = entry ? gtk_editable_get_text(GTK_EDITABLE(entry)) : "";
        if (!text || !*text) {
            query_timer = 0;
            apply_query("");
            return;
        }
        query_timer = g_timeout_add(kQueryDebounceMs, [](gpointer data) -> gboolean {
            auto *self = static_cast<Controller *>(data);
            self->query_timer = 0;
            const char *now = gtk_editable_get_text(GTK_EDITABLE(self->entry));
            self->apply_query(now ? now : "");
            return G_SOURCE_REMOVE;
        }, this);
    }

    void apply_query(const std::string &text)
    {
        query = text;
        query_gen.fetch_add(1);
        clear_heatmap();
        if (query.empty()) {
            if (index_ready) set_status(index_status, false);
            return;
        }
        if (!index_ready) return;
        run_query();
    }

    void run_query()
    {
        uint64_t qgen = query_gen.load();
        uint64_t bgen = build_gen.load();
        std::string q = query;
        worker.post([this, q, qgen, bgen] {
            if (qgen != query_gen.load() || bgen != build_gen.load() || !provider || !index) return;
            std::vector<float> qvec;
            std::string error;
            if (!provider->embed_batch({q}, qvec, error, {})) {
                if (qgen != query_gen.load() || bgen != build_gen.load()) return;
                post_main([this, qgen, bgen, error] {
                    if (qgen != query_gen.load() || bgen != build_gen.load()) return;
                    set_status(error.empty() ? "Query could not be embedded" : error, false);
                });
                return;
            }
            size_t n = index->count();
            std::vector<SemanticHit> hits(n);
            size_t got = n ? index->score_all(qvec.data(), hits.data(), n) : 0;
            hits.resize(got);
            post_main([this, qgen, bgen, hits = std::move(hits), got, n] {
                if (qgen != query_gen.load() || bgen != build_gen.load()) return;
                if (got != n) {
                    clear_heatmap();
                    set_status("Could not score sentence embeddings", false);
                    return;
                }
                last_hits = hits;
                paint(last_hits);
                set_status(index_status, false);
            });
        });
    }

    void schedule_rebuild()
    {
        if (rebuild_timer) g_source_remove(rebuild_timer);
        rebuild_timer = g_timeout_add(kRebuildDebounceMs, [](gpointer data) -> gboolean {
            auto *self = static_cast<Controller *>(data);
            self->rebuild_timer = 0;
            self->rebuild_now();
            return G_SOURCE_REMOVE;
        }, this);
    }

    void rebuild_now()
    {
        if (!open || !sci || !doc) return;
        uint64_t gen = build_gen.fetch_add(1) + 1;
        index_ready = false;
        clear_heatmap();
        sptr_t doc_len = send(sci, SCI_GETLENGTH, 0, 0);
        int codepage = static_cast<int>(send(sci, SCI_GETCODEPAGE, 0, 0));
        if (doc_len > kMaxBytes) {
            set_status("Document too large for semantic search", false);
            return;
        }
        if (codepage != 0 && codepage != SC_CP_UTF8) {
            set_status("Document is not valid UTF-8", false);
            return;
        }
        const char *chars = reinterpret_cast<const char *>(send(sci, SCI_GETCHARACTERPOINTER, 0, 0));
        std::string bytes = (chars && doc_len > 0) ? std::string(chars, static_cast<size_t>(doc_len))
                                                    : std::string();
        if (!bytes.empty() && !g_utf8_validate(bytes.data(), static_cast<gssize>(bytes.size()), nullptr)) {
            set_status("Document is not valid UTF-8", false);
            return;
        }
        int64_t first = next_id;
        std::string q = query;
        uint64_t qgen = query_gen.load();
        std::string want_model = model_id;
        std::string want_ep = ep_id;
        set_status("Indexing...", true);
        worker.post([this, gen, bytes = std::move(bytes), first, q, qgen, want_model, want_ep] {
            build_job(gen, bytes, first, q, qgen, want_model, want_ep);
        });
    }

    bool ensure_provider(const std::string &want_model, const std::string &want_ep, std::string &error)
    {
        const ModelSpec *spec = zoo.find(want_model);
        if (!spec) {
            error = "Unknown semantic model";
            return false;
        }
        bool same = provider && provider->available() && loaded_model == spec->id && loaded_ep == want_ep;
        if (same) return true;
        std::string dir = storage_dir(spec->id);
        std::string model_path = dir + "/" + spec->model.filename;
        bool have_model = g_file_test(model_path.c_str(), G_FILE_TEST_IS_REGULAR);
        if (!have_model) {
            std::string name = spec->name;
            post_main([this, gen = build_gen.load(), name] {
                if (!open) return;
                set_status("Downloading " + name + "...", true);
            });
        }
        if (!zoo.ensure(*spec, dir, error)) return false;
        OnnxProviderConfig cfg;
        cfg.model_path = dir + "/" + spec->model.filename;
        cfg.vocab_path = dir + "/" + spec->vocab.filename;
        cfg.dimension = spec->dimension;
        cfg.max_tokens = spec->max_tokens;
        cfg.pooling = spec->pooling;
        cfg.execution_provider = want_ep;
        auto created = make_onnx_provider(cfg, error);
        if (!created) return false;
        if (loaded_model != spec->id || loaded_ep != want_ep) cache.clear();
        provider = std::move(created);
        loaded_model = spec->id;
        loaded_ep = want_ep;
        backend_name = provider->backend_name();
        if (!engine) engine = make_cpu_similarity_engine();
        index = make_memory_vector_index(*engine);
        return true;
    }

    void build_job(uint64_t gen, const std::string &bytes, int64_t first,
                   const std::string &q, uint64_t qgen,
                   const std::string &want_model, const std::string &want_ep)
    {
        if (gen != build_gen.load()) return;
        auto spans_now = split_sentences(bytes);
        if (spans_now.size() > kMaxSentences) {
            post_main([this, gen] {
                if (gen != build_gen.load()) return;
                set_status("Document has too many sentences for semantic search", false);
            });
            return;
        }
        std::vector<Span> light;
        light.reserve(spans_now.size());
        for (const SentenceSpan &s : spans_now) {
            Span span;
            span.byte_start = s.byte_start;
            span.byte_length = s.byte_length;
            light.push_back(span);
        }
        post_main([this, gen, first, light] {
            if (gen != build_gen.load()) return;
            spans = light;
            first_id = first;
        });
        if (spans_now.empty()) {
            post_main([this, gen] {
                if (gen != build_gen.load()) return;
                next_id = first_id;
                index_ready = true;
                index_status = "No sentences to search";
                set_status(index_status, false);
            });
            return;
        }
        std::string error;
        if (!ensure_provider(want_model, want_ep, error)) {
            if (gen != build_gen.load()) return;
            post_main([this, gen, error] {
                if (gen != build_gen.load()) return;
                set_status(error.empty() ? "Could not load the embedding model" : error, false);
            });
            return;
        }
        if (gen != build_gen.load()) return;
        int dim = provider->dimension();
        index->reset(static_cast<size_t>(dim));

        std::vector<float> qvec;
        bool have_q = false;
        if (!q.empty() && qgen == query_gen.load()) {
            std::string qerr;
            if (provider->embed_batch({q}, qvec, qerr, {}) &&
                static_cast<int>(qvec.size()) == dim)
                have_q = true;
        }

        struct Row {
            std::string text;
            bool added = false;
        };
        std::vector<Row> rows(spans_now.size());
        std::vector<std::string> miss_text;
        std::vector<size_t> miss_row;
        size_t embedded = 0;
        auto add_row = [&](size_t row, const std::vector<float> &vec) {
            if (rows[row].added) return;
            if (static_cast<int>(vec.size()) != dim) return;
            if (!index->add(vec.data(), first + static_cast<int64_t>(row))) return;
            rows[row].added = true;
            ++embedded;
        };
        for (size_t i = 0; i < spans_now.size(); ++i) {
            rows[i].text = spans_now[i].text;
            auto found = cache.find(rows[i].text);
            if (found != cache.end() && static_cast<int>(found->second.size()) == dim) {
                add_row(i, found->second);
                continue;
            }
            bool queued = false;
            for (size_t m = 0; m < miss_text.size(); ++m) {
                if (miss_text[m] == rows[i].text) {
                    queued = true;
                    break;
                }
            }
            if (!queued) {
                miss_text.push_back(rows[i].text);
                miss_row.push_back(i);
            }
        }

        auto publish = [&](bool done) {
            if (gen != build_gen.load()) return;
            std::vector<SemanticHit> hits;
            if (have_q && qgen == query_gen.load() && index->count()) {
                hits.resize(index->count());
                size_t got = index->score_all(qvec.data(), hits.data(), hits.size());
                hits.resize(got);
            }
            size_t total = spans_now.size();
            size_t n_embedded = embedded;
            std::string backend = provider->backend_name();
            post_main([this, gen, hits = std::move(hits), done, total, n_embedded, backend, first] {
                if (gen != build_gen.load()) return;
                next_id = first + static_cast<int64_t>(total);
                if (!hits.empty()) {
                    last_hits = hits;
                    paint(hits);
                }
                char buf[256];
                if (done) {
                    std::string model_label = loaded_model;
                    if (const ModelSpec *spec = zoo.find(loaded_model)) {
                        if (!spec->name.empty()) model_label = spec->name;
                    }
                    index_status = std::to_string(n_embedded) + " of " +
                                   std::to_string(total) +
                                   " sentences indexed · " + model_label +
                                   " · " + backend;
                    index_ready = true;
                    set_status(index_status, false);
                    if (!query.empty()) run_query();
                } else {
                    std::snprintf(buf, sizeof buf, "Indexing %u of %u...",
                                  static_cast<unsigned>(n_embedded),
                                  static_cast<unsigned>(total));
                    set_status(buf, true);
                }
            });
        };

        if (!miss_text.empty()) {
            std::vector<float> out;
            std::string emb_error;
            bool ok = provider->embed_batch(miss_text, out, emb_error, [&](const std::vector<size_t> &done_idx) {
                if (gen != build_gen.load()) return false;
                for (size_t di : done_idx) {
                    if (di >= miss_text.size()) continue;
                    const float *src = out.data() + di * static_cast<size_t>(dim);
                    std::vector<float> vec(src, src + dim);
                    cache[miss_text[di]] = vec;
                    for (size_t row = 0; row < rows.size(); ++row) {
                        if (!rows[row].added && rows[row].text == miss_text[di])
                            add_row(row, vec);
                    }
                }
                publish(false);
                return gen == build_gen.load();
            });
            if (!ok) {
                if (gen != build_gen.load() || emb_error == "cancelled") return;
                post_main([this, gen, emb_error] {
                    if (gen != build_gen.load()) return;
                    set_status(emb_error.empty() ? "Could not embed sentences" : emb_error, false);
                });
                return;
            }
            if (cache.size() > kCacheLimit) cache.clear();
        }
        if (gen != build_gen.load()) return;
        publish(true);
    }

    void attach(GtkWidget *next)
    {
        NppDoc *next_doc = next ? static_cast<NppDoc *>(g_object_get_data(G_OBJECT(next), "npp-doc"))
                                : nullptr;
        if (next == sci && next_doc == doc) {
            if (open && !index_ready) rebuild_now();
            return;
        }
        build_gen.fetch_add(1);
        query_gen.fetch_add(1);
        if (rebuild_timer) {
            g_source_remove(rebuild_timer);
            rebuild_timer = 0;
        }
        clear_heatmap();
        sci = next;
        doc = next_doc;
        index_ready = false;
        spans.clear();
        if (!sci || !doc) return;
        configure_indicator();
        rebuild_now();
    }

    void close_bar()
    {
        open = false;
        if (query_timer) g_source_remove(query_timer);
        if (rebuild_timer) g_source_remove(rebuild_timer);
        query_timer = 0;
        rebuild_timer = 0;
        build_gen.fetch_add(1);
        query_gen.fetch_add(1);
        clear_heatmap();
        index_ready = false;
        sci = nullptr;
        doc = nullptr;
        if (revealer) gtk_revealer_set_reveal_child(GTK_REVEALER(revealer), FALSE);
        GtkWidget *cur = nullptr;
        NppDoc *current = editor_current_doc();
        if (current) cur = current->sci;
        if (cur) gtk_widget_grab_focus(cur);
    }
};

Controller &ctrl()
{
    static Controller c;
    return c;
}

void on_entry_changed(GtkEditable *, gpointer)
{
    if (!ctrl().ui_ready) return;
    ctrl().schedule_query();
}

gboolean on_entry_key(GtkEventControllerKey *, guint keyval, guint, GdkModifierType, gpointer)
{
    if (keyval == GDK_KEY_Escape) {
        ctrl().close_bar();
        return TRUE;
    }
    if (keyval == GDK_KEY_Return || keyval == GDK_KEY_KP_Enter) {
        if (ctrl().query_timer) g_source_remove(ctrl().query_timer);
        ctrl().query_timer = 0;
        const char *text = gtk_editable_get_text(GTK_EDITABLE(ctrl().entry));
        ctrl().apply_query(text ? text : "");
        return TRUE;
    }
    return FALSE;
}

void on_sensitivity(GObject *, GParamSpec *, gpointer)
{
    if (!ctrl().ui_ready || !ctrl().sens_dd) return;
    guint sel = gtk_drop_down_get_selected(GTK_DROP_DOWN(ctrl().sens_dd));
    if (sel > 2) return;
    ctrl().sensitivity = static_cast<int>(sel) - 1;
    ctrl().save_prefs();
    ctrl().paint(ctrl().last_hits);
}

void on_model(GObject *, GParamSpec *, gpointer)
{
    if (!ctrl().ui_ready || !ctrl().model_dd) return;
    guint sel = gtk_drop_down_get_selected(GTK_DROP_DOWN(ctrl().model_dd));
    if (sel >= ctrl().model_ids.size()) return;
    if (ctrl().model_ids[sel] == ctrl().model_id) return;
    ctrl().model_id = ctrl().model_ids[sel];
    ctrl().refresh_legend();
    ctrl().save_prefs();
    if (ctrl().open) ctrl().rebuild_now();
}

void on_ep(GObject *, GParamSpec *, gpointer)
{
    if (!ctrl().ui_ready || !ctrl().ep_dd) return;
    guint sel = gtk_drop_down_get_selected(GTK_DROP_DOWN(ctrl().ep_dd));
    if (sel >= ctrl().ep_ids.size()) return;
    if (ctrl().ep_ids[sel] == ctrl().ep_id) return;
    ctrl().ep_id = ctrl().ep_ids[sel];
    ctrl().save_prefs();
    if (ctrl().open) ctrl().rebuild_now();
}

void on_swatch(GtkToggleButton *button, gpointer)
{
    if (!ctrl().ui_ready) return;
    int band = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button), "band"));
    if (band < 0 || band > 5) return;
    if (gtk_toggle_button_get_active(button)) ctrl().band_mask |= (1u << band);
    else ctrl().band_mask &= ~(1u << band);
    ctrl().paint(ctrl().last_hits);
}

void on_close_clicked(GtkButton *, gpointer) { ctrl().close_bar(); }

} /* namespace */

GtkWidget *semantic_search_bar_new(void)
{
    Controller &c = ctrl();
    std::string manifest_error;
    gchar *manifest = npp_bundle_file("semantic", "models.json");
    if (!c.zoo.load_file(manifest, manifest_error))
        c.index_status = manifest_error.empty() ? "Semantic model list is missing" : manifest_error;
    g_free(manifest);
    c.load_prefs();
    if (c.model_id.empty() || !c.zoo.find(c.model_id))
        c.model_id = c.zoo.default_id();

    c.revealer = gtk_revealer_new();
    gtk_revealer_set_transition_type(GTK_REVEALER(c.revealer), GTK_REVEALER_TRANSITION_TYPE_SLIDE_UP);
    gtk_revealer_set_reveal_child(GTK_REVEALER(c.revealer), FALSE);

    GtkWidget *outer = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_widget_set_margin_start(outer, 8);
    gtk_widget_set_margin_end(outer, 8);
    gtk_widget_set_margin_top(outer, 4);
    gtk_widget_set_margin_bottom(outer, 4);

    GtkWidget *row1 = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget *row2 = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_append(GTK_BOX(outer), row1);
    gtk_box_append(GTK_BOX(outer), row2);

    GtkWidget *label = gtk_label_new("Semantic:");
    c.entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(c.entry), "Describe what you're looking for…");
    gtk_accessible_update_property(GTK_ACCESSIBLE(c.entry),
                                   GTK_ACCESSIBLE_PROPERTY_LABEL,
                                   "Semantic search query", -1);
    gtk_widget_set_hexpand(c.entry, TRUE);
    g_signal_connect(c.entry, "changed", G_CALLBACK(on_entry_changed), nullptr);
    GtkEventController *keys = gtk_event_controller_key_new();
    g_signal_connect(keys, "key-pressed", G_CALLBACK(on_entry_key), nullptr);
    gtk_widget_add_controller(c.entry, keys);

    GtkStringList *models = gtk_string_list_new(nullptr);
    for (const ModelSpec &spec : c.zoo.models()) {
        gtk_string_list_append(models, spec.name.c_str());
        c.model_ids.push_back(spec.id);
    }
    c.model_dd = gtk_drop_down_new(G_LIST_MODEL(models), nullptr);
    gtk_widget_set_tooltip_text(c.model_dd, "Sentence embedding model");

    GtkStringList *eps = gtk_string_list_new(nullptr);
    for (const ExecProviderInfo &ep : execution_provider_catalog()) {
        std::string label_ep = ep.id;
        for (char &ch : label_ep)
            if (ch >= 'a' && ch <= 'z') ch = static_cast<char>(ch - 'a' + 'A');
        if (ep.id == "cpu") label_ep = "CPU";
        if (ep.id == "auto") label_ep = "Automatic";
        gtk_string_list_append(eps, label_ep.c_str());
        c.ep_ids.push_back(ep.id);
    }
    if (c.ep_ids.empty()) {
        gtk_string_list_append(eps, "CPU");
        c.ep_ids.push_back("cpu");
    }
    c.ep_dd = gtk_drop_down_new(G_LIST_MODEL(eps), nullptr);
    gtk_widget_set_tooltip_text(c.ep_dd,
        "Execution provider. A missing GPU provider falls back to CPU.");

    GtkWidget *close = gtk_button_new_with_label("x");
    gtk_widget_set_tooltip_text(close, "Close semantic search");
    g_signal_connect(close, "clicked", G_CALLBACK(on_close_clicked), nullptr);

    gtk_box_append(GTK_BOX(row1), label);
    gtk_box_append(GTK_BOX(row1), c.entry);
    gtk_box_append(GTK_BOX(row1), c.model_dd);
    gtk_box_append(GTK_BOX(row1), c.ep_dd);
    gtk_box_append(GTK_BOX(row1), close);

    GtkStringList *sens = gtk_string_list_new(nullptr);
    gtk_string_list_append(sens, "Strict");
    gtk_string_list_append(sens, "Standard");
    gtk_string_list_append(sens, "Broad");
    c.sens_dd = gtk_drop_down_new(G_LIST_MODEL(sens), nullptr);
    gtk_widget_set_tooltip_text(c.sens_dd,
        "Heatmap sensitivity. Broad paints weaker matches green. Strict requires stronger matches. Scores stay the same.");

    GtkWidget *swatches = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    for (int i = 0; i < 6; ++i) {
        c.swatch[i] = gtk_toggle_button_new();
        char cls[32];
        std::snprintf(cls, sizeof cls, "sem-band-%d", i);
        gtk_widget_add_css_class(c.swatch[i], cls);
        gtk_widget_set_tooltip_text(c.swatch[i],
            "Show only the selected similarity bands and bookmark their lines. Click again to deselect. Several bands can be selected.");
        char alabel[64];
        std::snprintf(alabel, sizeof alabel, "Similarity band %d of 6", i + 1);
        gtk_accessible_update_property(GTK_ACCESSIBLE(c.swatch[i]),
                                       GTK_ACCESSIBLE_PROPERTY_LABEL, alabel, -1);
        g_object_set_data(G_OBJECT(c.swatch[i]), "band", GINT_TO_POINTER(i));
        g_signal_connect(c.swatch[i], "toggled", G_CALLBACK(on_swatch), nullptr);
        gtk_box_append(GTK_BOX(swatches), c.swatch[i]);
    }
    c.legend_css = gtk_css_provider_new();
    gtk_style_context_add_provider_for_display(
        gdk_display_get_default(), GTK_STYLE_PROVIDER(c.legend_css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    // The display owns the provider for the application lifetime.
    g_object_unref(c.legend_css);
    c.refresh_legend();

    c.spinner = gtk_spinner_new();
    c.status = gtk_label_new("");
    gtk_label_set_ellipsize(GTK_LABEL(c.status), PANGO_ELLIPSIZE_END);
    gtk_label_set_xalign(GTK_LABEL(c.status), 0.f);
    gtk_widget_set_hexpand(c.status, TRUE);

    gtk_box_append(GTK_BOX(row2), c.sens_dd);
    gtk_box_append(GTK_BOX(row2), swatches);
    gtk_box_append(GTK_BOX(row2), c.spinner);
    gtk_box_append(GTK_BOX(row2), c.status);
    gtk_revealer_set_child(GTK_REVEALER(c.revealer), outer);

    guint sens_index = static_cast<guint>(c.sensitivity + 1);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(c.sens_dd), sens_index);
    guint model_index = 0;
    for (guint i = 0; i < c.model_ids.size(); ++i)
        if (c.model_ids[i] == c.model_id) model_index = i;
    if (!c.model_ids.empty())
        gtk_drop_down_set_selected(GTK_DROP_DOWN(c.model_dd), model_index);
    guint ep_index = 0;
    bool ep_ok = false;
    for (guint i = 0; i < c.ep_ids.size(); ++i) {
        if (c.ep_ids[i] == c.ep_id) {
            ep_index = i;
            ep_ok = true;
        }
    }
    if (!ep_ok) c.ep_id = c.ep_ids.empty() ? "cpu" : c.ep_ids[0];
    gtk_drop_down_set_selected(GTK_DROP_DOWN(c.ep_dd), ep_index);

    g_signal_connect(c.sens_dd, "notify::selected", G_CALLBACK(on_sensitivity), nullptr);
    g_signal_connect(c.model_dd, "notify::selected", G_CALLBACK(on_model), nullptr);
    g_signal_connect(c.ep_dd, "notify::selected", G_CALLBACK(on_ep), nullptr);
    c.ui_ready = true;
    if (!c.index_status.empty()) c.set_status(c.index_status, false);
    return c.revealer;
}

void semantic_search_toggle(void)
{
    Controller &c = ctrl();
    if (!c.revealer) return;
    if (c.open) {
        c.close_bar();
        return;
    }
    c.open = true;
    gtk_revealer_set_reveal_child(GTK_REVEALER(c.revealer), TRUE);
    gtk_widget_grab_focus(c.entry);
    NppDoc *doc = editor_current_doc();
    c.attach(doc ? doc->sci : nullptr);
}

void semantic_search_note_modified(GtkWidget *sci)
{
    Controller &c = ctrl();
    if (!c.open || !sci || sci != c.sci) return;
    c.index_ready = false;
    c.build_gen.fetch_add(1);
    c.clear_heatmap();
    c.set_status("Indexing...", true);
    c.schedule_rebuild();
}

void semantic_search_note_activated(GtkWidget *sci)
{
    Controller &c = ctrl();
    if (!c.open) return;
    c.attach(sci);
}

void semantic_search_note_closed(NppDoc *doc)
{
    if (!doc) return;
    Controller &c = ctrl();
    if (c.doc == doc) {
        c.build_gen.fetch_add(1);
        c.query_gen.fetch_add(1);
        /* Clear while sci is still alive, so bookmarks this feature
         * added are removed before the map entry goes away. */
        c.clear_heatmap();
        c.sci = nullptr;
        c.doc = nullptr;
        c.index_ready = false;
        c.spans.clear();
    }
    c.our_bookmarks.erase(doc);
}
