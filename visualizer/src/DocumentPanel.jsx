import { useMemo } from 'react';

const PERSON_LABELS = new Set(['person_ref', 'person_name', 'person_role']);

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
    if (!text) return [{ id: 'empty', text: '', isEntity: false }];
    if (!spans.length) return [{ id: 'raw-text', text, isEntity: false }];

    const sorted = [...spans].sort((a, b) => a.start - b.start);
    const result = [];
    let lastEnd = 0;

    sorted.forEach((span, idx) => {
      if (span.start > lastEnd) {
        result.push({ id: `txt-${idx}`, text: text.substring(lastEnd, span.start), isEntity: false });
      }
      result.push({
        id: span.id,
        text: text.substring(span.start, span.end) || span.text || 'N/A',
        isEntity: true,
        span
      });
      lastEnd = span.end;
    });

    if (lastEnd < text.length) {
      result.push({ id: 'txt-end', text: text.substring(lastEnd), isEntity: false });
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

          const isHovered = hoveredSpanId === chunk.id;
          const isLinked = getLinkedIds(chunk.id).includes(hoveredSpanId) || getLinkedIds(hoveredSpanId).includes(chunk.id);
          const style = getEntityStyle(chunk.span.label);

          let className = 'entity-span';
          if (isHovered) className += ' is-hovered';
          if (isLinked) className += ' is-linked';

          return (
            <span
              key={chunk.id}
              className={className}
              style={{
                backgroundColor: isHovered ? '#ffc9c9' : style.bg,
                borderBottomColor: style.border,
              }}
              onMouseEnter={() => setHoveredSpanId(chunk.id)}
              onMouseLeave={() => setHoveredSpanId(null)}
              title={`Label: ${chunk.span.label}\nID: ${chunk.id}`}
            >
              {chunk.text}
            </span>
          );
        })}
      </div>
    </div>
  );
}
