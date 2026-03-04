(function () {
  const API_BASE = "http://127.0.0.1:8000";
  const REFRESH_MS = 60_000;

  function parseSymbol() {
    const fromQuery = new URL(window.location.href).searchParams.get("symbol");
    if (fromQuery) {
      const cleaned = decodeURIComponent(fromQuery);
      return cleaned.includes(":") ? cleaned.split(":")[1] : cleaned;
    }

    const pathParts = window.location.pathname.split("/").filter(Boolean);
    const last = pathParts[pathParts.length - 1] || "";
    if (last.includes(":")) return last.split(":")[1];
    return "AAPL";
  }

  function ensureBadge() {
    let badge = document.getElementById("tv-ai-prediction-badge");
    if (!badge) {
      badge = document.createElement("div");
      badge.id = "tv-ai-prediction-badge";
      badge.textContent = "AI prediction: loading...";
      document.body.appendChild(badge);
    }
    return badge;
  }

  async function updateBadge() {
    const badge = ensureBadge();
    const symbol = parseSymbol();
    try {
      const res = await fetch(`${API_BASE}/predict/${symbol}`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      badge.textContent = `AI next close (${data.symbol}): ${Number(data.predicted_next_close).toFixed(2)}`;
      badge.dataset.state = "ok";
    } catch (error) {
      badge.textContent = `AI prediction unavailable (${symbol})`;
      badge.dataset.state = "error";
    }
  }

  updateBadge();
  setInterval(updateBadge, REFRESH_MS);
})();
