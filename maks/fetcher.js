(async () => {
  const START_PAGE = 1;   // ← resume from last checkpoint
  const MAX_PAGE   = 297;
  const PER_PAGE   = 50;
  const BASE       = 'https://findit.dtu.dk';
  const SAVE_EVERY = 10;

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // Separate strategies: 429 = exponential (3s base), network = linear (5s flat)
  const fetchWithBackoff = async (url, attempt = 0) => {
    const MAX_ATTEMPTS = 10;
    try {
      const res = await fetch(url, {credentials: 'include'});
      if (res.status === 429) {
        if (attempt >= MAX_ATTEMPTS) throw new Error('Max retries (429)');
        const wait = 3000 * Math.pow(2, attempt) + Math.random() * 1000; // 3 → 6 → 12 → 24s
        console.warn(`⏳ 429 — waiting ${(wait/1000).toFixed(1)}s (attempt ${attempt+1})`);
        await sleep(wait);
        return fetchWithBackoff(url, attempt + 1);
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res;
    } catch(e) {
      if (e.message.startsWith('Max retries')) throw e;
      if (attempt >= MAX_ATTEMPTS) throw new Error(`Max retries (network): ${e.message}`);
      // Network drop: flat 5s retry, don't exponentially back off
      const wait = 5000 + Math.random() * 2000;
      console.warn(`⏳ Network drop — retrying in ${(wait/1000).toFixed(1)}s (attempt ${attempt+1})`);
      await sleep(wait);
      return fetchWithBackoff(url, attempt + 1);
    }
  };

  const extractEntries = (doc) => {
    const entries = [];
    doc.querySelectorAll('.js-access').forEach(div => {
      const resolverUrl = div.getAttribute('data-url');
      if (!resolverUrl) return;
      const card = div.closest('div.result');
      entries.push({
        id:          card?.dataset?.id ?? div.getAttribute('data-id') ?? null,
        title:       card?.querySelector('[itemprop="name"],.result__title,h3,h2,h1')?.textContent?.trim() ?? null,
        year:        card?.querySelector('[itemprop="datePublished"],.year,time')?.textContent?.trim() ?? null,
        author:      card?.querySelector('[itemprop="author"],.result__creator,.creator')?.textContent?.trim() ?? null,
        resolverUrl
      });
    });
    return entries;
  };

  const saveFiles = (results, label) => {
    const ja = document.createElement('a');
    ja.href = URL.createObjectURL(new Blob([JSON.stringify(results, null, 2)], {type:'application/json'}));
    ja.download = `dtu_theses_${label}.json`;
    document.body.appendChild(ja); ja.click(); document.body.removeChild(ja);

    const ta = document.createElement('a');
    ta.href = URL.createObjectURL(new Blob([results.map(r=>r.resolverUrl).join('\n')], {type:'text/plain'}));
    ta.download = `dtu_theses_${label}.txt`;
    document.body.appendChild(ta); ta.click(); document.body.removeChild(ta);
    console.log(`💾 Saved: dtu_theses_${label}.json + .txt`);
  };

  const allResults = [];
  let consecutiveErrors = 0;
  let page = START_PAGE;
  let nextUrl = `/en/catalog?from=2015&page=${START_PAGE}&per_page=${PER_PAGE}&sort=year&to=2025&type=thesis_master`;

  console.log(`▶️ Resuming from page ${START_PAGE}...`);
  console.time('Elapsed');

  while (nextUrl && page <= MAX_PAGE) {
    const fullUrl = nextUrl.startsWith('http') ? nextUrl : BASE + nextUrl;
    let doc;
    try {
      const res = await fetchWithBackoff(fullUrl);
      doc = new DOMParser().parseFromString(await res.text(), 'text/html');
      consecutiveErrors = 0;
    } catch(e) {
      console.error(`❌ Page ${page} failed: ${e.message}`);
      if (++consecutiveErrors >= 3) {
        console.error('3 consecutive failures — saving and stopping.');
        break;
      }
      // Skip this page and continue
      page++;
      nextUrl = `/en/catalog?from=2015&page=${page}&per_page=${PER_PAGE}&sort=year&to=2025&type=thesis_master`;
      continue;
    }

    const entries = extractEntries(doc);
    allResults.push(...entries);

    const nextEl = doc.querySelector('a.pager__next');
    nextUrl = nextEl ? nextEl.getAttribute('href') : null;

    const pct = ((page / MAX_PAGE) * 100).toFixed(1);
    console.log(`📄 Page ${page}/${MAX_PAGE} (${pct}%) +${entries.length} → total: ${allResults.length}`);

    if (page % SAVE_EVERY === 0) saveFiles(allResults, `checkpoint_p${page}`);

    page++;
    await sleep(2500 + Math.random() * 1000);
  }

  console.timeEnd('Elapsed');
  console.log(`✅ Done! ${allResults.length} entries from pages ${START_PAGE}–${page-1}`);
  saveFiles(allResults, `p${START_PAGE}_to_${page-1}`);
})();