function parseCSV(text) {
  if (!text) return [];
  const [header, ...lines] = text.trim().replace(/\r/g, '').split('\n');
  const keys = header.split(',');
  return lines.filter(l => l.trim()).map(line => {
    const vals = line.split(',');
    return Object.fromEntries(keys.map((k, i) => [k.trim(), (vals[i] || '').trim()]));
  });
}

function normalizeSpans(data) {
  if (!data) return [];
  const needs = (data.needs || []).map(s => ({ ...s, type: 'need' }));
  const persons = (data.persons || []).map(s => ({ ...s, type: 'person' }));
  return [...needs, ...persons].sort((a, b) => a.start - b.start);
}

function buildRelationMap(spans, relations) {
  const map = {};
  spans.forEach(s => { map[s.id] = { linked: [] }; });
  (relations || []).forEach(r => {
    const source = r.from || r[0];
    const target = r.to || r[1];
    if (map[source]) map[source].linked.push(target);
    if (map[target]) map[target].linked.push(source);
  });
  return map;
}

export { parseCSV, normalizeSpans, buildRelationMap };