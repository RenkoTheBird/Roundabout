import { useEffect, useMemo, useState } from "react";
import { runClaimDetectionInBrowser } from "./claimModel.js";
import { sourceChecker, getSourceDate, getSourceName } from "../search/source-quality.js";
import { EXACTITUDE_THRESHOLD } from "../search/exactitude.js";
import './App.css';

function platformLabel(platform) {
  switch (platform) {
    case "reddit":
      return "Reddit";
    case "twitter":
      return "Twitter";
    case "instagram":
      return "Instagram";
    default:
      return "—";
  }
}

function App() {
  const [post, setPost] = useState(null);
  const [claimsStatus, setClaimsStatus] = useState("loading");
  const [claimsResult, setClaimsResult] = useState(null);
  const [selectedClaimIndex, setSelectedClaimIndex] = useState(0);
  const [resultsByClaimIndex, setResultsByClaimIndex] = useState({});
  const [expandedSourceIndex, setExpandedSourceIndex] = useState(null);
  const [sourceScoresByClaimIndex, setSourceScoresByClaimIndex] = useState({});
  const [sourceNamesByClaimIndex, setSourceNamesByClaimIndex] = useState({});
  const [debugOpen, setDebugOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [debugTab, setDebugTab] = useState("post");

  useEffect(() => {
    let currentTabId = null;
    let storageListener = null;

    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (!tab?.id) return;

      currentTabId = tab.id;

      chrome.runtime.sendMessage(
        { type: "get-current-post", tabId: currentTabId },
        (response) => {
          if (response) {
            setPost(response);
          }
        }
      );

      const storageKey = `post_${currentTabId}`;
      storageListener = (changes, areaName) => {
        if (areaName === "local" && changes[storageKey]) {
          setPost(changes[storageKey].newValue);
        }
      };

      chrome.storage.onChanged.addListener(storageListener);

      setClaimsStatus("loading");
      setClaimsResult(null);
      setResultsByClaimIndex({});
      setSelectedClaimIndex(0);
      chrome.runtime.sendMessage(
        { type: "get-claims", tabId: currentTabId },
        async (response) => {
          if (response?.error) {
            setClaimsResult({ clauses: [], detectedClaims: [], clauseDecisions: [], error: response.error });
            setClaimsStatus("result");
            return;
          }
          const clauses = response?.clauses ?? [];
          try {
            const clauseDecisions = await runClaimDetectionInBrowser(clauses);
            const detectedClaims = clauseDecisions.filter((d) => d.isClaimFinal).map((d) => d.clause);
            setClaimsResult({ clauses, detectedClaims, clauseDecisions });
            setClaimsStatus("result");
          } catch (err) {
            setClaimsResult({ clauses, detectedClaims: [], clauseDecisions: [], error: err?.message || "Model failed to run" });
            setClaimsStatus("result");
          }
        }
      );
    });

    return () => {
      if (storageListener) {
        chrome.storage.onChanged.removeListener(storageListener);
      }
    };
  }, []);

  const clauseDecisions = Array.isArray(claimsResult?.clauseDecisions) ? claimsResult.clauseDecisions : [];
  const claims = useMemo(
    () => clauseDecisions.filter((d) => d.isClaimFinal),
    [clauseDecisions]
  );
  useEffect(() => {
    if (claims.length > 0 && selectedClaimIndex >= claims.length) {
      setSelectedClaimIndex(0);
    }
  }, [claims.length, selectedClaimIndex]);

  useEffect(() => {
    setExpandedSourceIndex(null);
  }, [selectedClaimIndex]);

  useEffect(() => {
    const claimList = claims;
    if (claimList.length === 0 || selectedClaimIndex < 0 || selectedClaimIndex >= claimList.length) return;
    const cached = resultsByClaimIndex[selectedClaimIndex];
    if (cached !== undefined) return;

    const query = claimList[selectedClaimIndex].clause;
    setResultsByClaimIndex((prev) => ({ ...prev, [selectedClaimIndex]: { loading: true } }));

    chrome.runtime.sendMessage({ type: "search-claim", query }, (response) => {
      if (chrome.runtime.lastError) {
        setResultsByClaimIndex((prev) => ({
          ...prev,
          [selectedClaimIndex]: { error: chrome.runtime.lastError.message },
        }));
        return;
      }
      if (response?.error) {
        setResultsByClaimIndex((prev) => ({
          ...prev,
          [selectedClaimIndex]: { error: response.error },
        }));
        return;
      }
      setResultsByClaimIndex((prev) => ({
        ...prev,
        [selectedClaimIndex]: { results: response?.results ?? [] },
      }));
    });
  }, [claims, selectedClaimIndex, resultsByClaimIndex]);

  useEffect(() => {
    const claimList = claims;
    const results = resultsByClaimIndex[selectedClaimIndex]?.results;
    const claim = claimList[selectedClaimIndex]?.clause;
    if (!results?.length || !claim) return;

    let cancelled = false;
    (async () => {
      try {
        const scores = await Promise.all(results.map((r) => sourceChecker(claim, r, post ?? undefined)));
        if (!cancelled) {
          setSourceScoresByClaimIndex((prev) => ({ ...prev, [selectedClaimIndex]: scores }));
        }
      } catch {
        if (!cancelled) {
          setSourceScoresByClaimIndex((prev) => ({ ...prev, [selectedClaimIndex]: [] }));
        }
      }
    })();
    return () => { cancelled = true; };
  }, [claims, selectedClaimIndex, resultsByClaimIndex, post]);

  useEffect(() => {
    const results = resultsByClaimIndex[selectedClaimIndex]?.results;
    if (!results?.length) return;

    let cancelled = false;
    (async () => {
      const names = await Promise.all(results.map((r) => getSourceName(r?.url)));
      if (!cancelled) {
        setSourceNamesByClaimIndex((prev) => ({ ...prev, [selectedClaimIndex]: names }));
      }
    })();
    return () => { cancelled = true; };
  }, [selectedClaimIndex, resultsByClaimIndex]);

  const clauses = Array.isArray(claimsResult?.clauses) ? claimsResult.clauses : [];
  const errorMsg = claimsResult?.error;
  const selectedCache = resultsByClaimIndex[selectedClaimIndex];

  return (
    <div className="popup-shell">
      {helpOpen && <HelpPanel />}
      {debugOpen && (
        <DebugPanel
          post={post}
          claimsStatus={claimsStatus}
          clauses={clauses}
          clauseDecisions={clauseDecisions}
          errorMsg={errorMsg}
          debugTab={debugTab}
          setDebugTab={setDebugTab}
        />
      )}

      <header className="popup-header">
        <span className="popup-header__brand">Roundabout</span>
        <span className="popup-header__site">{platformLabel(post?.platform)}</span>
        <div className="popup-header__actions">
          <button
            type="button"
            className="popup-header__secondary-btn"
            onClick={() => {
              setHelpOpen((v) => !v);
              if (!helpOpen) setDebugOpen(false);
            }}
            aria-expanded={helpOpen}
          >
            Help
          </button>
          <button
            type="button"
            className="popup-header__secondary-btn"
            onClick={() => {
              setDebugOpen((v) => !v);
              if (!debugOpen) setHelpOpen(false);
            }}
            aria-expanded={debugOpen}
          >
            Debug
          </button>
        </div>
      </header>

      <main className="popup-main">
        {!post ? (
          <div className="empty-state bodyFont">No post detected</div>
        ) : (
          <>
            <section className="claims-section panel">
              {claimsStatus === "loading" && (
                <p className="bodyFont claims-loading">Loading...</p>
              )}
              {claimsStatus === "result" && (
                <>
                  {errorMsg && (
                    <p className="bodyFont claims-error">Could not run detection: {errorMsg}</p>
                  )}
                  {!errorMsg && claims.length === 0 && (
                    <p className="bodyFont claims-none">No claims detected.</p>
                  )}
                  {!errorMsg && claims.length > 0 && (
                    <ul className="claims-list">
                      {claims.map((claim, i) => (
                        <li key={i}>
                          <button
                            type="button"
                            className={`claim-item claim-item-btn ui-font ${i === selectedClaimIndex ? "claim-item--selected" : ""}`}
                            onClick={() => setSelectedClaimIndex(i)}
                          >
                            <div className="claim-row-main">{claim.clause}</div>
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </section>

            {claims.length > 0 && (
              <section className="search-section">
                <h2 className="section-title ui-font">Search results</h2>
                <div className="panel">
                  <p className="bodyFont search-results-hint">Click a claim above to see its search results.</p>
                  {selectedCache?.loading && (
                    <p className="bodyFont claims-loading">Loading...</p>
                  )}
                  {selectedCache?.error && (
                    <p className="bodyFont claims-error">{selectedCache.error}</p>
                  )}
                  {selectedCache?.results && selectedCache.results.length === 0 && (
                    <p className="bodyFont claims-none">No results.</p>
                  )}
                  {selectedCache?.results && selectedCache.results.length > 0 && (
                    <ul className="claims-list search-results-list">
                      {(() => {
                        const scores = sourceScoresByClaimIndex[selectedClaimIndex] || [];
                        const items = selectedCache.results.map((r, i) => ({
                          result: r,
                          originalIndex: i,
                          score: scores[i] ?? null,
                        }));
                        items.sort((a, b) => {
                          if (a.score == null && b.score == null) return 0;
                          if (a.score == null) return 1;
                          if (b.score == null) return -1;
                          return b.score - a.score;
                        });
                        const names = sourceNamesByClaimIndex[selectedClaimIndex] || [];
                        return items.map((item, displayIndex) => {
                          const sourceDate = getSourceDate(item.result);
                          const sourceName = names[item.originalIndex] ?? null;
                          return (
                            <li key={item.originalIndex} className="search-result-item">
                              <button
                                type="button"
                                className="search-result-header ui-font"
                                onClick={() => setExpandedSourceIndex(expandedSourceIndex === item.originalIndex ? null : item.originalIndex)}
                              >
                                <span className="source-number">Source {displayIndex + 1}</span>
                                <span className="search-result-title">{item.result.title ?? ""}</span>
                                {sourceName && (
                                  <span className="source-name-label bodyFont">{sourceName}</span>
                                )}
                                <span className="source-score-box" aria-label={`Source score ${item.score != null ? item.score : "loading"}`}>
                                  {item.score != null ? Math.round(item.score) : "…"}
                                </span>
                              </button>
                              {expandedSourceIndex === item.originalIndex && (
                                <div className="search-result-details">
                                  {sourceDate != null && (
                                    <p className="bodyFont search-result-date">Date: {sourceDate.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" })}</p>
                                  )}
                                  <a href={item.result.url} target="_blank" rel="noopener noreferrer" className="bodyFont search-result-url">{item.result.url ?? ""}</a>
                                  <div className="bodyFont search-result-desc">{item.result.description ?? ""}</div>
                                </div>
                              )}
                            </li>
                          );
                        });
                      })()}
                    </ul>
                  )}
                </div>
              </section>
            )}

            {claimsStatus === "result" && !errorMsg && claims.length === 0 && (
              <p className="bodyFont claims-hint claims-hint--below">Search runs for detected claims only.</p>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function HelpPanel() {
  return (
    <div className="help-panel" role="region" aria-label="Help: scores and detection">
      <h2 className="help-panel__title ui-font">About scores &amp; detection</h2>
      <div className="help-panel__scroll bodyFont">
        <section className="help-section">
          <h3 className="help-section__title ui-font">Exactitude</h3>
          <p>
            Exactitude measures how concrete and checkable a sentence is. Sub-scores <strong>A</strong> through{" "}
            <strong>F</strong> each range from 0 to 2. A seventh component, <strong>G</strong> (personal relativity),
            only reduces the total when the wording is personal or experiential. Everything is summed and capped so the
            total stays between 0 and 12. A sentence is only listed as a <strong>claim</strong> here if its Exactitude
            is at least {EXACTITUDE_THRESHOLD} <em>and</em> the LR step below agrees it is claim-like.
          </p>
        </section>
        <section className="help-section">
          <h3 className="help-section__title ui-font">Exactitude sub-metrics (A–F)</h3>
          <dl className="help-dl">
            <dt className="help-dt">A — Quantification</dt>
            <dd className="help-dd">
              Numbers, money, percentages, amounts, and comparison language (e.g. “more than”, “the same as”) make a
              statement easier to verify or pin down.
            </dd>
            <dt className="help-dt">B — Time specificity</dt>
            <dd className="help-dd">
              Explicit dates or clock times score highest; looser words like “recently” or “today” add a weaker signal.
            </dd>
            <dt className="help-dt">C — Location scope</dt>
            <dd className="help-dd">
              Places, facilities, and organizations that anchor where something applies.
            </dd>
            <dt className="help-dt">D — Defined terms</dt>
            <dd className="help-dd">
              How clearly the sentence identifies what is being discussed (concrete noun phrases rather than vague
              wording).
            </dd>
            <dt className="help-dt">E — Source clarity</dt>
            <dd className="help-dd">
              Phrases that attribute information (“according to…”, “published by…”) and clear references to sources.
            </dd>
            <dt className="help-dt">F — Falsifiability</dt>
            <dd className="help-dd">
              Whether the phrasing could be checked or argued against. Heavy use of modals (“might”, “could”) or
              subjective hedges (“I think”, “probably”) lowers this score; factual verb use with concrete anchors raises
              it.
            </dd>
          </dl>
          <p className="help-section__note">
            <strong>G</strong> (personal relativity) appears in Debug together with A–F; it penalizes first-person and
            similar framing. Numeric breakdowns for each sentence are under <strong>Debug → Exactitude / LR</strong>.
          </p>
        </section>
        <section className="help-section">
          <h3 className="help-section__title ui-font">LR and the LR score</h3>
          <p>
            <strong>LR</strong> means <strong>logistic regression</strong>. Each sentence is converted to a numeric
            vector using a small on-device language model (MiniLM). A trained model turns that vector into an{" "}
            <strong>LR score</strong> (a real number, not limited to 0–1). If the score is{" "}
            <strong>zero or positive</strong>, the extension treats the sentence as claim-like at this step. The final
            claim list uses{" "}
            <strong>both</strong> this LR decision and Exactitude (see above).
          </p>
        </section>
        <section className="help-section">
          <h3 className="help-section__title ui-font">NER</h3>
          <p>
            <strong>NER</strong> is <strong>named entity recognition</strong>: an on-device model labels spans of text
            (people, places, dates, amounts, etc.). You may see labels such as{" "}
            <strong>PER</strong> (person), <strong>ORG</strong> (organization), <strong>GPE</strong>,{" "}
            <strong>LOC</strong>, and <strong>FAC</strong> (locations and facilities), <strong>DATE</strong> and{" "}
            <strong>TIME</strong>,{" "}
            <strong>MONEY</strong>, <strong>PERCENT</strong>, <strong>QUANTITY</strong>, <strong>CARDINAL</strong>,{" "}
            <strong>MISC</strong>, and <strong>WORK_OF_ART</strong>. Roundabout feeds these signals into Exactitude. The
            exact entities found for each sentence are shown under <strong>Debug → Exactitude / LR</strong>.
          </p>
        </section>
      </div>
    </div>
  );
}

function DebugPanel({
  post,
  claimsStatus,
  clauses,
  clauseDecisions,
  errorMsg,
  debugTab,
  setDebugTab,
}) {
  return (
    <div className="debug-panel">
      <div className="debug-tabs" role="tablist" aria-label="Debug sections">
        <button
          type="button"
          role="tab"
          aria-selected={debugTab === "post"}
          className={`debug-tab ${debugTab === "post" ? "debug-tab--active" : ""}`}
          onClick={() => setDebugTab("post")}
        >
          Post
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={debugTab === "sentences"}
          className={`debug-tab ${debugTab === "sentences" ? "debug-tab--active" : ""}`}
          onClick={() => setDebugTab("sentences")}
        >
          Sentences checked
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={debugTab === "scores"}
          className={`debug-tab ${debugTab === "scores" ? "debug-tab--active" : ""}`}
          onClick={() => setDebugTab("scores")}
        >
          Exactitude / LR
        </button>
      </div>
      <div className="debug-tab-content">
        {debugTab === "post" && (
          <div className="debug-scroll bodyFont">
            {post ? (
              <PlatformPost post={post} />
            ) : (
              <p className="claims-none">No post loaded.</p>
            )}
          </div>
        )}
        {debugTab === "sentences" && (
          <div className="debug-scroll bodyFont">
            {claimsStatus === "loading" && (
              <p className="claims-loading">Loading...</p>
            )}
            {claimsStatus === "result" && errorMsg && (
              <p className="claims-error">Could not load clauses: {errorMsg}</p>
            )}
            {claimsStatus === "result" && !errorMsg && clauses.length === 0 && (
              <p className="claims-none">No clauses (post too short or no sentence/clause breaks).</p>
            )}
            {claimsStatus === "result" && !errorMsg && clauses.length > 0 && (
              <ol className="sentences-checked-list">
                {clauses.map((c, i) => (
                  <li key={i} className="sentence-checked-item">{c}</li>
                ))}
              </ol>
            )}
          </div>
        )}
        {debugTab === "scores" && (
          <div className="debug-scroll bodyFont">
            {claimsStatus === "loading" && (
              <p className="claims-loading">Loading...</p>
            )}
            {claimsStatus === "result" && errorMsg && (
              <p className="claims-error">Could not run detection: {errorMsg}</p>
            )}
            {claimsStatus === "result" && !errorMsg && clauseDecisions.length === 0 && (
              <p className="claims-none">No scoring data.</p>
            )}
            {claimsStatus === "result" && !errorMsg && clauseDecisions.length > 0 && (
              <ul className="scores-debug-list">
                {clauseDecisions.map((dec, i) => (
                  <li key={i} className="scores-debug-item">
                    <div className="claim-row-main ui-font">{dec.clause}</div>
                    <div className="claim-row-metrics claim-row-metrics--accent">
                      Exactitude: {dec.exactitudeTotal}/12
                      {" | "}
                      A:{dec.exactitudeBreakdown.quantification}
                      {" B:"}{dec.exactitudeBreakdown.timeSpecificity}
                      {" C:"}{dec.exactitudeBreakdown.locationScope}
                      {" D:"}{dec.exactitudeBreakdown.definedTerms}
                      {" E:"}{dec.exactitudeBreakdown.sourceClarity}
                      {" F:"}{dec.exactitudeBreakdown.falsifiability}
                      {" G:"}{dec.exactitudeBreakdown.personalRelativity ?? 0}
                      {" | "}
                      LR: {dec.lrScore.toFixed(2)}
                      {" | "}
                      Claim: {dec.isClaimFinal ? "yes" : "no"}
                    </div>
                    <NerEntitiesLine entities={dec.nerEntities} />
                    <pre className="claim-debug-pre">{JSON.stringify(dec.exactitudeSignals, null, 2)}</pre>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** NER entity labels from Xenova/bert-base-NER (same pipeline as exactitude.js). */
function NerEntitiesLine({ entities }) {
  const list = Array.isArray(entities) ? entities : [];
  return (
    <div className="claim-row-ner bodyFont" aria-label="NER entity types for this sentence">
      <span className="claim-row-ner-label">NER:</span>{" "}
      {list.length === 0 ? (
        <span className="ner-entities-empty">(no entities)</span>
      ) : (
        list.map((e, idx) => (
          <span key={idx}>
            {idx > 0 ? " · " : null}
            <span className="ner-entity-label">{e.label}</span>
            <span className="ner-entity-colon">: </span>
            <span className="ner-entity-text">{e.text}</span>
          </span>
        ))
      )}
    </div>
  );
}

function PlatformPost({ post }) {
  switch (post.platform) {
    case "reddit":
      return (
        <>
          <h3 className="platform-post-title ui-font">{post.title}</h3>
          <p className="bodyFont platform-post-body">{post.body}</p>
        </>
      );
    case "instagram":
      return <p className="bodyFont platform-post-body">{post.caption}</p>;
    case "twitter":
      return <p className="bodyFont platform-post-body">{post.text}</p>;
    default:
      return <p className="bodyFont">Unsupported post</p>;
  }
}

export default App;
