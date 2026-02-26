import { useEffect, useState } from "react";
import { runClaimDetectionInBrowser } from "./claimModel.js";
import { sourceChecker } from "../search/source-quality.js";
import './App.css';

function App() {
  const [post, setPost] = useState(null);
  const [claimsStatus, setClaimsStatus] = useState("loading");
  const [claimsResult, setClaimsResult] = useState(null);
  const [selectedClaimIndex, setSelectedClaimIndex] = useState(0);
  const [resultsByClaimIndex, setResultsByClaimIndex] = useState({});
  const [expandedSourceIndex, setExpandedSourceIndex] = useState(null);
  const [sourceScoresByClaimIndex, setSourceScoresByClaimIndex] = useState({});

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
            setClaimsResult({ clauses: [], detectedClaims: [], error: response.error });
            setClaimsStatus("result");
            return;
          }
          const clauses = response?.clauses ?? [];
          try {
            const detectedClaims = await runClaimDetectionInBrowser(clauses);
            setClaimsResult({ clauses, detectedClaims });
            setClaimsStatus("result");
          } catch (err) {
            setClaimsResult({ clauses, detectedClaims: [], error: err?.message || "Model failed to run" });
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

  const claims = Array.isArray(claimsResult?.detectedClaims) ? claimsResult.detectedClaims : [];

  // When claims load with at least one claim, ensure we have selectedClaimIndex in range
  useEffect(() => {
    if (claims.length > 0 && selectedClaimIndex >= claims.length) {
      setSelectedClaimIndex(0);
    }
  }, [claims.length, selectedClaimIndex]);

  // Collapse source details when switching to another claim
  useEffect(() => {
    setExpandedSourceIndex(null);
  }, [selectedClaimIndex]);

  // Trigger Brave search for the selected claim when it has no cached result
  useEffect(() => {
    const claimList = Array.isArray(claimsResult?.detectedClaims) ? claimsResult.detectedClaims : [];
    if (claimList.length === 0 || selectedClaimIndex < 0 || selectedClaimIndex >= claimList.length) return;
    const cached = resultsByClaimIndex[selectedClaimIndex];
    if (cached !== undefined) return;

    const query = claimList[selectedClaimIndex];
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
  }, [claimsResult?.detectedClaims, selectedClaimIndex, resultsByClaimIndex]);

  // Compute source scores when we have results and a selected claim
  useEffect(() => {
    const claimList = Array.isArray(claimsResult?.detectedClaims) ? claimsResult.detectedClaims : [];
    const results = resultsByClaimIndex[selectedClaimIndex]?.results;
    const claim = claimList[selectedClaimIndex];
    if (!results?.length || !claim) return;

    let cancelled = false;
    (async () => {
      const scores = await Promise.all(results.map((r) => sourceChecker(claim, r)));
      if (!cancelled) {
        setSourceScoresByClaimIndex((prev) => ({ ...prev, [selectedClaimIndex]: scores }));
      }
    })();
    return () => { cancelled = true; };
  }, [claimsResult?.detectedClaims, selectedClaimIndex, resultsByClaimIndex]);

  if (!post) {
    return <div className="postFont">No post detected</div>;
  }

  const clauses = Array.isArray(claimsResult?.clauses) ? claimsResult.clauses : [];
  const errorMsg = claimsResult?.error;
  const selectedCache = resultsByClaimIndex[selectedClaimIndex];
  const selectedClaim = claims[selectedClaimIndex];

  return (
    <>
      <h1>Roundabout</h1>
      <h2 className="postFont">Current post</h2>
      <div className="panel">
        <PlatformPost post={post} />
      </div>
      <h2 className="postFont" style={{ marginTop: "16px" }}>Clauses checked</h2>
      <div className="panel">
        {claimsStatus === "loading" ? (
          <p className="postFont claims-loading">Loading...</p>
        ) : clauses.length === 0 ? (
          <p className="postFont claims-none">No clauses (post too short or no sentence/clause breaks).</p>
        ) : (
          <ul className="claims-list">
            {clauses.map((c, i) => (
              <li key={i} className="postFont claim-item">{c}</li>
            ))}
          </ul>
        )}
      </div>
      <h2 className="postFont" style={{ marginTop: "16px" }}>Detected claims</h2>
      <div className="panel">
        {claimsStatus === "loading" && (
          <p className="postFont claims-loading">Loading...</p>
        )}
        {claimsStatus === "result" && (
          <>
            <p className="postFont claims-result-label">Result</p>
            {errorMsg && (
              <p className="postFont claims-error">Could not run detection: {errorMsg}</p>
            )}
            {!errorMsg && (
              claims.length === 0 ? (
                <p className="postFont claims-none">No claims detected.</p>
              ) : (
                <ul className="claims-list">
                  {claims.map((claim, i) => (
                    <li key={i}>
                      <button
                        type="button"
                        className={`postFont claim-item claim-item-btn ${i === selectedClaimIndex ? "claim-item--selected" : ""}`}
                        onClick={() => setSelectedClaimIndex(i)}
                      >
                        {claim}
                      </button>
                    </li>
                  ))}
                </ul>
              )
            )}
          </>
        )}
      </div>
      {claims.length > 0 && (
        <>
          <h2 className="postFont" style={{ marginTop: "16px" }}>Search results</h2>
          <div className="panel">
            <p className="postFont search-results-hint">Click a claim above to see its search results.</p>
            {selectedCache?.loading && (
              <p className="postFont claims-loading">Loading...</p>
            )}
            {selectedCache?.error && (
              <p className="postFont claims-error">{selectedCache.error}</p>
            )}
            {selectedCache?.results && selectedCache.results.length === 0 && (
              <p className="postFont claims-none">No results.</p>
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
                  return items.map((item, displayIndex) => (
                  <li key={item.originalIndex} className="search-result-item">
                    <button
                      type="button"
                      className="search-result-header"
                      onClick={() => setExpandedSourceIndex(expandedSourceIndex === item.originalIndex ? null : item.originalIndex)}
                    >
                      <span className="source-number">Source {displayIndex + 1}</span>
                      <span className="postFont search-result-title">{item.result.title ?? ""}</span>
                      {(() => {
                        const score = item.score;
                        return (
                          <span className="source-score-box" aria-label={`Source score ${score != null ? score : "loading"}`}>
                            {score != null ? Math.round(score) : "…"}
                          </span>
                        );
                      })()}
                    </button>
                    {expandedSourceIndex === item.originalIndex && (
                      <div className="search-result-details">
                        <a href={item.result.url} target="_blank" rel="noopener noreferrer" className="postFont search-result-url">{item.result.url ?? ""}</a>
                        <div className="postFont search-result-desc">{item.result.description ?? ""}</div>
                      </div>
                    )}
                  </li>
                  ));
                })()}
              </ul>
            )}
          </div>
        </>
      )}
      {claimsStatus === "result" && !errorMsg && claims.length === 0 && (
        <p className="postFont claims-hint" style={{ marginTop: "8px" }}>Search runs for detected claims only.</p>
      )}
    </>
  );
}

function PlatformPost({ post }) {
  switch (post.platform) {
    case "reddit":
      return (
        <>
          <h3 className="postFont">{post.title}</h3>
          <p className="postFont">{post.body}</p>
        </>
      );
    case "instagram":
      return <p className="postFont">{post.caption}</p>;
    case "twitter":
      return <p className="postFont">{post.text}</p>;
    default:
      return <p className="postFont">Unsupported post</p>;
  }
}

export default App;

