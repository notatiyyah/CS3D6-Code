import { useMemo } from 'react';

const PERSON_LABELS = new Set(['person_name', 'person_role']);

const getEntityStyle = (label) => {
  if (PERSON_LABELS.has(label)) {
    return { bg: '#d0ebff', border: '#1c7ed6' };
  }
  return { bg: '#fff3bf', border: '#e67700' };
};

export default function DocumentPanel({ title, spans, relationMap, text, hoveredSpanId, setHoveredSpanId }) {
  const getLinkedIds = (id) => {
    if (!id || !relationMap[id]) return [];
    return relationMap[id].linked;
  };

  const chunks = useMemo(() => {
    if (!text) return [{ id: 'empty', text: '', isEntity: false, spans: [] }];
    if (!spans.length) return [{ id: 'raw-text', text, isEntity: false, spans: [] }];

    // 1. Collect every unique boundary point (start or end) across all spans.
    const boundarySet = new Set([0, text.length]);
    spans.forEach((s) => {
      boundarySet.add(s.start);
      boundarySet.add(s.end);
    });
    const boundaries = [...boundarySet].sort((a, b) => a - b);

    // 2. Walk consecutive boundary pairs; each pair is an atomic segment
    //    that either has zero or more spans covering it in full.
    const result = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
      const segStart = boundaries[i];
      const segEnd = boundaries[i + 1];
      if (segStart === segEnd) continue;

      const covering = spans
        .filter((s) => s.start <= segStart && s.end >= segEnd)
        // most specific (shortest) span first — used as the "primary" id for hover
        .sort((a, b) => (a.end - a.start) - (b.end - b.start));

      result.push({
        id: `seg-${segStart}-${segEnd}`,
        text: text.substring(segStart, segEnd),
        isEntity: covering.length > 0,
        spans: covering
      });
    }
    return result;
  }, [text, spans]);

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>{title}</h2>
        <span className="count-badge">{spans.length} entities</span>
      </div>
      <div className="document-view">
        {chunks.map((chunk) => {
          if (!chunk.isEntity) {
            return <span key={chunk.id}>{chunk.text}</span>;
          }

          const coveringIds = chunk.spans.map((s) => s.id);
          const isHovered = coveringIds.includes(hoveredSpanId);
          const isLinked = coveringIds.some(
            (id) => getLinkedIds(id).includes(hoveredSpanId) || getLinkedIds(hoveredSpanId).includes(id)
          );
          const isOverlap = chunk.spans.length > 1;

          // Primary span (shortest = most specific) drives the hover target & fill color.
          const primary = chunk.spans[0];
          const primaryStyle = getEntityStyle(primary.label);

          let className = 'entity-span';
          if (isHovered) className += ' is-hovered';
          if (isLinked) className += ' is-linked';
          if (isOverlap) className += ' is-overlap';

          // Stack one underline bar per covering span so overlap depth is visible at a glance.
          const boxShadow = chunk.spans
            .map((s, i) => `0 ${2 + i * 3}px 0 0 ${getEntityStyle(s.label).border}`)
            .join(', ');

          const tooltip = chunk.spans
            .map((s) => `Label: ${s.label}\nID: ${s.id}`)
            .join('\n---\n');

          return (
            <span
              key={chunk.id}
              className={className}
              style={{
                backgroundColor: isHovered ? '#ffc9c9' : primaryStyle.bg,
                backgroundImage: isOverlap && !isHovered
                  ? `repeating-linear-gradient(45deg, transparent 0 4px, rgba(0,0,0,0.08) 4px 8px)`
                  : 'none',
                boxShadow,
                paddingBottom: `${4 + chunk.spans.length * 3}px`,
              }}
              onMouseEnter={() => setHoveredSpanId(primary.id)}
              onMouseLeave={() => setHoveredSpanId(null)}
              title={tooltip}
            >
              {chunk.text}
            </span>
          );
        })}
      </div>
    </div>
  );
}