// Search Brave API for given claim
const BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search";

/**
 * @param {string} query - Search query (claim text)
 * @param {string} apiKey - Brave API subscription token
 * @returns {Promise<{ web: { results: Array<{ title?: string; url?: string; description?: string }> } }>}
 * @throws or returns structure with error on failure
 */
export async function searchBrave(query, apiKey) {
  if (!query || typeof query !== "string") {
    return { error: "Invalid query" };
  }
  const url = new URL(BRAVE_SEARCH_URL);
  url.searchParams.set("q", query.trim());

  const headers = {
    "X-Subscription-Token": apiKey,
  };

  const response = await fetch(url.toString(), { headers });

  if (!response.ok) {
    const text = await response.text();
    return {
      error: `Brave API error: ${response.status} ${response.statusText}${text ? ` - ${text.slice(0, 200)}` : ""}`,
    };
  }

  const data = await response.json();
  if (data.web && Array.isArray(data.web.results)) {
    return { results: data.web.results };
  }
  return { results: [] };
}
