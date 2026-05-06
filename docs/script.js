(() => {
  const $ = (id) => document.getElementById(id);

  const ui = {
    random: $("random"),
    category: $("category"),
    theme: $("theme"),
    result: $("result"),
    error: $("error"),
    thumb: $("thumb"),
    themeLabel: $("theme-label"),
    categoryLabel: $("category-label"),
    tv: $("tv"), pv: $("pv"), ev: $("ev"),
    ta: $("ta"), pa: $("pa"), ea: $("ea"),
    bins: $("bins"),
  };

  let data = null;
  let filtered = [];
  let byTheme = {};

  const fmt = (n, d = 3) => n.toFixed(d);

  function showError(msg) {
    ui.error.textContent = msg;
    ui.error.hidden = false;
    ui.result.hidden = true;
  }

  function populateCategoryFilter(categories) {
    for (const cat of categories) {
      const opt = document.createElement("option");
      opt.value = cat;
      opt.textContent = cat;
      ui.category.appendChild(opt);
    }
  }

  function populateThemeSelect(items) {
    ui.theme.innerHTML = "";
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item.theme;
      opt.textContent = item.theme;
      ui.theme.appendChild(opt);
    }
  }

  function applyCategoryFilter() {
    const cat = ui.category.value;
    filtered = cat
      ? data.items.filter((d) => d.category === cat)
      : data.items.slice();
    populateThemeSelect(filtered);
    if (filtered.length) {
      ui.theme.value = filtered[0].theme;
      render(byTheme[filtered[0].theme]);
    }
  }

  function pickRandom() {
    if (!filtered.length) return;
    const item = filtered[Math.floor(Math.random() * filtered.length)];
    ui.theme.value = item.theme;
    render(item);
  }

  function renderBins(item) {
    const { bins, bin_hex } = data;
    ui.bins.innerHTML = "";
    // Sort bins by composition descending so the dominant ones surface first.
    const order = bins
      .map((name, i) => ({ name, i, pct: item.composition[i], dom: item.dominance[i] }))
      .sort((a, b) => b.pct - a.pct);
    for (const { name, i, pct, dom } of order) {
      const li = document.createElement("li");
      const label = document.createElement("span");
      label.className = "label" + (dom ? " dominant" : "");
      label.textContent = name;

      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${(pct * 100).toFixed(1)}%`;
      fill.style.background = bin_hex[i];
      bar.appendChild(fill);

      const pctSpan = document.createElement("span");
      pctSpan.className = "pct";
      pctSpan.textContent = `${(pct * 100).toFixed(1)}%`;

      li.append(label, bar, pctSpan);
      ui.bins.appendChild(li);
    }
  }

  function render(item) {
    if (!item) return;
    ui.result.hidden = false;
    ui.thumb.src = encodeURI(item.thumb);
    ui.thumb.alt = item.theme;
    ui.themeLabel.textContent = item.theme;
    ui.categoryLabel.textContent = item.category;

    ui.tv.textContent = fmt(item.true_v);
    ui.pv.textContent = fmt(item.pred_v);
    ui.ev.textContent = fmt(item.se_v, 4);
    ui.ta.textContent = fmt(item.true_a);
    ui.pa.textContent = fmt(item.pred_a);
    ui.ea.textContent = fmt(item.se_a, 4);

    renderBins(item);
  }

  fetch("predictions.json")
    .then((r) => {
      if (!r.ok) throw new Error(`predictions.json: ${r.status}`);
      return r.json();
    })
    .then((payload) => {
      data = payload;
      byTheme = Object.fromEntries(payload.items.map((d) => [d.theme, d]));
      populateCategoryFilter(payload.categories);
      applyCategoryFilter();

      ui.category.addEventListener("change", applyCategoryFilter);
      ui.theme.addEventListener("change", () => render(byTheme[ui.theme.value]));
      ui.random.addEventListener("click", pickRandom);
    })
    .catch((err) => {
      showError(`Failed to load predictions.json — ${err.message}`);
    });
})();
