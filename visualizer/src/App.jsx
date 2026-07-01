import { useState, useEffect, useMemo } from 'react';
import './App.css';

// --- 1. Data Normalization Helper ---
function normalizeData(data) {
  let entitiesMap = {};
  let entityArray = [];
  
  let rawEntities = [...(data.needs || []), ...(data.persons || [])];
  
  rawEntities.forEach(e => {
    let ent = {
      id: e.id,
      text: e.text || data.text?.substring(e.start, e.end) || "N/A",
      start: e.start,
      end: e.end,
      label: e.label,
      score: e.score || null
    };
    entitiesMap[e.id] = ent;
    entityArray.push(ent);
  });

  // Sort by start index so items within their groups flow naturally
  entityArray.sort((a, b) => a.start - b.start);

  let relations = [];
  if (data.relations) {
    data.relations.forEach(r => {
      if (Array.isArray(r)) { 
        relations.push({ from: r[0], to: r[1] });
      } else { 
        relations.push({ from: r.from, to: r.to });
      }
    });
  }
  return { text: data.text, entitiesMap, entityArray, relations };
}

// --- 2. Sub-components ---

const TextRenderer = ({ text, activeSpans }) => {
  if (!text) return <div className="text-box"></div>;

  const elements = [];
  for (let i = 0; i < text.length; i++) {
    let char = text[i];
    
    let appliedClasses = activeSpans
      .filter(s => i >= s.start && i < s.end)
      .map(s => s.cls)
      .join(' ');

    if (appliedClasses) {
      elements.push(<span key={i} className={appliedClasses}>{char}</span>);
    } else {
      elements.push(char);
    }
  }

  return <div className="text-box">{elements}</div>;
};

const DebuggerColumn = ({ title, data }) => {
  const [activeSpans, setActiveSpans] = useState([]);
  
  if (!data) return <div className="column"><h2>{title}</h2><p>Loading...</p></div>;

  const { text, entityArray, relations, entitiesMap } = useMemo(() => normalizeData(data), [data]);
  const knownLabels = ['person_ref', 'care_care_setting', 'care_social_care_involvement'];

  // Split entities into groups for easier comparison
  const persons = entityArray.filter(ent => ent.label === 'person_ref');
  const needs = entityArray.filter(ent => ent.label !== 'person_ref');

  // Helper function to keep the JSX clean
  const renderEntityList = (list) => {
    if (list.length === 0) return <p className="empty-state" style={{ fontSize: '14px', color: '#888' }}>None found.</p>;
    
    return list.map((ent, idx) => (
      <div 
        key={`${ent.id}-${idx}`} 
        className="item"
        onMouseEnter={() => setActiveSpans([{ start: ent.start, end: ent.end, cls: 'hl-primary' }])}
        onMouseLeave={() => setActiveSpans([])}
      >
        <span className={`badge ${knownLabels.includes(ent.label) ? ent.label : 'default-label'}`}>
          {ent.label}
        </span>
        <strong>{ent.text}</strong>
        <span style={{ color: '#aaa', fontSize: '12px', marginLeft: '6px' }}>
          [{ent.start}, {ent.end}]
        </span>
        {ent.score && <span className="score">{(ent.score * 100).toFixed(1)}%</span>}
      </div>
    ));
  };

  return (
    <div className="column">
      <h2>{title}</h2>
      
      <TextRenderer text={text} activeSpans={activeSpans} />
      
      <h3>Needs & Care Settings</h3>
      <div>
        {renderEntityList(needs)}
      </div>

      <h3>Person References</h3>
      <div>
        {renderEntityList(persons)}
      </div>

      <h3>Relations</h3>
      <div>
        {relations.length === 0 && <p className="empty-state" style={{ fontSize: '14px', color: '#888' }}>No relations found.</p>}
        {relations.map((rel, idx) => {
          const fromEnt = entitiesMap[rel.from];
          const toEnt = entitiesMap[rel.to];
          
          if (!fromEnt || !toEnt) return null;

          return (
            <div 
              key={idx} 
              className="item"
              onMouseEnter={() => setActiveSpans([
                { start: fromEnt.start, end: fromEnt.end, cls: 'hl-primary' },
                { start: toEnt.start, end: toEnt.end, cls: 'hl-secondary' }
              ])}
              onMouseLeave={() => setActiveSpans([])}
            >
              <strong>{fromEnt.text}</strong> 
              <span className="rel-arrow">➔</span> 
              <strong>{toEnt.text}</strong>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// --- 3. Main App Component ---

export default function App() {
  const [gtDataset, setGtDataset] = useState([]);
  const [mlDataset, setMlDataset] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [status, setStatus] = useState('loading'); // loading, error, ready

  useEffect(() => {
    async function fetchData() {
      try {
        // Fetches from the public folder
        const [gtRes, mlRes] = await Promise.all([
          fetch('/data/val_data.json'),
          fetch('/data/e2e.span_model_relation_model.json')
        ]);

        if (!gtRes.ok || !mlRes.ok) throw new Error("Failed to load JSON files.");

        const gtData = await gtRes.json();
        const mlData = await mlRes.json();

        setGtDataset(gtData);
        setMlDataset(mlData);
        setStatus('ready');
      } catch (err) {
        console.error(err);
        setStatus('error');
      }
    }
    fetchData();
  }, []);

  const maxRecords = Math.min(gtDataset.length, mlDataset.length);

  return (
    <div>
      <div className="header">
        <h1>NER & Relation Debugger</h1>
        
        {status === 'error' && (
          <p className="error">Error loading data. Did you move the <b>data</b> folder into the <b>public</b> folder?</p>
        )}

        {status === 'ready' && (
          <div className="controls" style={{ display: 'flex', justifyContent: 'center', gap: '15px', marginTop: '15px', alignItems: 'center' }}>
            <button 
              onClick={() => setCurrentIndex(c => Math.max(0, c - 1))}
              disabled={currentIndex === 0}
            >
              &#8592; Prev
            </button>
            
            <select 
              value={currentIndex} 
              onChange={(e) => setCurrentIndex(Number(e.target.value))}
              style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc', minWidth: '400px' }}
            >
              {Array.from({ length: maxRecords }).map((_, i) => (
                <option key={i} value={i}>
                  Record {i + 1} {gtDataset[i]?.id ? ` - ${gtDataset[i].id}` : ''}
                </option>
              ))}
            </select>
            
            <button 
              onClick={() => setCurrentIndex(c => Math.min(maxRecords - 1, c + 1))}
              disabled={currentIndex >= maxRecords - 1}
            >
              Next &#8594;
            </button>
            
            <span style={{ fontWeight: 'bold', color: '#555', minWidth: '60px' }}>
              {currentIndex + 1} / {maxRecords}
            </span>
          </div>
        )}
      </div>

      {status === 'ready' && (
        <div className="container">
          <DebuggerColumn title="Ground Truth" data={gtDataset[currentIndex]} />
          <DebuggerColumn title="ML Model Prediction" data={mlDataset[currentIndex]} />
        </div>
      )}
    </div>
  );
}