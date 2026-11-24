"use strict";

const $ = (id) => document.getElementById(id);
const number = new Intl.NumberFormat();
const connection = $("connection");
const keyStates = new Map([
  ["active", ["Active", "status-active"]],
  ["cooldown", ["Cooldown", "status-waiting"]],
  ["quarantined", ["Quarantined", "status-blocked"]],
  ["budget_exceeded", ["Budget exceeded", "status-blocked"]],
  ["disabled", ["Disabled", "status-muted"]],
]);
let loading = false;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function tokens(value) {
  return value === null || value === undefined ? "—" : number.format(value);
}

function amount(value, currency) {
  return value === null || value === undefined ? "—" : `${value} ${currency}`;
}

function metric(label, value) {
  const tile = node("div", undefined, "metric");
  tile.append(node("strong", value), node("span", label));
  return tile;
}

function keyState(status) {
  const [label, style] = keyStates.get(status) || [status, "status-muted"];
  return node("span", label, `status-chip ${style}`);
}

function chart(container, rows, label, value, annotation) {
  if (!rows.length) {
    container.append(node("p", "No recorded usage", "empty"));
    return;
  }
  const max = Math.max(1e-12, ...rows.map(value));
  for (const row of rows) {
    const item = node("div", undefined, "bar");
    const title = node("div", undefined, "bar-label");
    title.append(node("span", label(row)), node("span", annotation(row)));
    const bar = node("progress");
    bar.max = max;
    bar.value = value(row);
    bar.setAttribute("aria-label", label(row));
    item.append(title, bar);
    container.append(item);
  }
}

function table(container, headings, rows) {
  container.replaceChildren();
  if (!rows.length) {
    container.append(node("p", "No records yet", "empty"));
    return;
  }
  const element = node("table");
  const heading = element.createTHead().insertRow();
  for (const label of headings) {
    const cell = node("th", label);
    cell.scope = "col";
    heading.append(cell);
  }
  const body = element.createTBody();
  for (const row of rows) {
    const line = body.insertRow();
    for (const value of row) {
      if (typeof value === "string") line.insertCell().textContent = value;
      else line.insertCell().append(value);
    }
  }
  container.append(element);
}

async function read(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("Dashboard data unavailable");
  return response.json();
}

function render(daily, models, keys, events) {
  const totals = $("totals");
  totals.replaceChildren(
    metric("Attempts · 14 days", tokens(models.reduce((sum, row) => sum + row.attempts, 0))),
    metric("Unpriced attempts · 14 days", tokens(models.reduce((sum, row) => sum + row.unpriced, 0))),
    metric("Active keys", tokens(keys.filter((key) => key.status === "active").length)),
  );
  const spend = $("spend");
  spend.replaceChildren();
  const currencies = [...new Set(daily.map((row) => row.currency))];
  if (!currencies.length) chart(spend, [], null, null, null);
  for (const currency of currencies) {
    spend.append(node("p", currency, "subhead"));
    chart(spend, daily.filter((row) => row.currency === currency),
      (row) => row.group, (row) => Number(row.cost), (row) => amount(row.cost, row.currency));
  }
  const tokenChart = $("tokens");
  tokenChart.replaceChildren();
  const modelTokens = new Map();
  for (const row of models) {
    modelTokens.set(row.group, (modelTokens.get(row.group) || 0) + row.input_tokens + row.output_tokens);
  }
  const top = [...modelTokens].sort((left, right) => right[1] - left[1]).slice(0, 8);
  chart(tokenChart, top, ([model]) => model, ([, count]) => count, ([, count]) => tokens(count));

  table($("keys"), ["Pool", "Key", "Status", "Selections", "Today", "Total", "Cooldown ends"],
    keys.map((key) => [
      key.pool_id, key.key_alias,
      keyState(key.status),
      tokens(key.selections), amount(key.daily_spend, key.currency),
      amount(key.total_spend, key.currency), key.cooldown_until || "—",
    ]));
  table($("models"), ["Model", "Input", "Output", "Cached", "Cache writes", "Spend", "Unpriced"],
    models.map((row) => [
      row.group, tokens(row.input_tokens), tokens(row.output_tokens), tokens(row.cached_input_tokens),
      tokens(row.cache_write_tokens), amount(row.cost, row.currency), tokens(row.unpriced),
    ]));
  table($("events"), ["Finished · UTC", "Key", "Model", "Tag", "HTTP", "Outcome", "Spend", "Pricing"],
    events.map((event) => [
      event.finished_at.replace("T", " ").slice(0, 19), `${event.pool_id}/${event.key_alias}`,
      event.model || "—", event.tag || "—", event.http_status?.toString() || "—", event.outcome,
      amount(event.cost, event.currency), event.pricing_status,
    ]));
}

async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const since = new Date(Date.now() - 13 * 86400_000).toISOString().slice(0, 10);
    const [daily, models, keys, events] = await Promise.all([
      read(`/api/stats?by=day&since=${since}`), read(`/api/stats?by=model&since=${since}`),
      read("/api/keys"), read("/api/events?limit=25"),
    ]);
    render(daily.rows, models.rows, keys.keys, events.events);
    connection.textContent = "Live";
    connection.dataset.state = "live";
  } catch {
    connection.textContent = "Unable to refresh";
    connection.dataset.state = "error";
  } finally {
    loading = false;
  }
}

$("refresh").addEventListener("click", refresh);
refresh();
setInterval(refresh, 10_000);
