import React, { useState, useEffect, useMemo } from 'react';
import './App.css';
import config from './config.js';
import { parseCSV, normalizeSpans, buildRelationMap } from './helpers.js'
import DocumentPanel from './DocumentPanel.jsx';

export default function App() {
    // Datasets
    const [gtDataset, setGtDataset] = useState([]);
    const [mlDataset, setMlDataset] = useState([]);
    const [elRows, setElRows] = useState([]);

    // Page status
    const [currentIndex, setCurrentIndex] = useState(0);
    const [status, setStatus] = useState('loading');
    const [hoveredSpanId, setHoveredSpanId] = useState(null);

    useEffect(() => {
        // Load data from files (ground truth, predictions, and the entity linking)
        async function fetchData() {
            try {
                const [gtRes, mlRes, elRes] = await Promise.all([
                    fetch(`/data/${config.groundTruthFile}`),
                    fetch(`/data/${config.predictionsFile}`),
                    fetch(`/data/${config.entityLinkingFile}`),
                ]);
                if (!gtRes.ok || !mlRes.ok) throw new Error('Failed to load datasets.');

                setGtDataset(await gtRes.json());
                setMlDataset(await mlRes.json());
                if (elRes.ok) setElRows(parseCSV(await elRes.text()));
                setStatus('ready');
            } catch (err) {
                console.error("Data loading failure:", err);
                setStatus('error');
            }
        }
        fetchData();
    }, []);

    const maxRecords = Math.min(gtDataset.length, mlDataset.length);
    const gtRecord = gtDataset[currentIndex];
    const mlRecord = mlDataset[currentIndex];

    const generateRecordSummary = useMemo(() => {
        // Compile ground truth, predictions, and entity linking into one dict for just the selected record
        if (!gtRecord || !mlRecord) return null;

        const gtSpans = normalizeSpans(gtRecord);
        const mlSpans = normalizeSpans(mlRecord);

        return {
            id: gtRecord.id,
            text: gtRecord.text,
            household_members: gtRecord.household_members || [],
            gt: {
                spans: gtSpans,
                relations: gtRecord.relations || [],
                relationMap: buildRelationMap(gtSpans, gtRecord.relations)
            },
            ml: {
                spans: mlSpans,
                relations: mlRecord.relations || [],
                relationMap: buildRelationMap(mlSpans, mlRecord.relations)
            },
            linking: elRows.filter(r => r.note_id === gtRecord.id)
        };
    }, [currentIndex, gtRecord, mlRecord, elRows]);

    const getPersonName = (id) => {
        const member = generateRecordSummary?.household_members?.find(m => m.id === id);
        return member ? member.fullName : id;
    };

    // ---------
    // Rendering
    // --------

    if (status === 'loading') return <div className="fallback-state">Loading...</div>;
    if (status === 'error' || maxRecords === 0) {
        return <div className="fallback-state error">Error loading data. Check that the data files are in the correct folder.</div>;
    }

    return (
        <div className="debugger-root">
            {/* Navigation Bar */}
            <header className="control-bar">
                <div className="brand">
                    <h1>Annotation Comparison Tool</h1>
                    <span className="version-tag">v1</span>
                </div>
                <div className="navigation">
                    <button onClick={() => setCurrentIndex(c => Math.max(0, c - 1))} disabled={currentIndex === 0}>
                        Prev
                    </button>
                    <select value={currentIndex} onChange={e => setCurrentIndex(Number(e.target.value))}>
                        {Array.from({ length: maxRecords }).map((_, i) => (
                            <option key={i} value={i}>
                                Record {i + 1} {gtDataset[i]?.id ? ` (${gtDataset[i].id.substring(0, 8)})` : ''}
                            </option>
                        ))}
                    </select>
                    <button onClick={() => setCurrentIndex(c => Math.min(maxRecords - 1, c + 1))} disabled={currentIndex >= maxRecords - 1}>
                        Next
                    </button>
                    <span className="index-tracker">{currentIndex + 1} / {maxRecords}</span>
                </div>
            </header>

            {generateRecordSummary && (
                <main className="workspace">
                    {/* Three-panel view */}
                    <div className="split-view gap-3-way">
                        {/* GT */}
                        <DocumentPanel
                            title="Ground Truth (Annotations)"
                            spans={generateRecordSummary.gt.spans}
                            relationMap={generateRecordSummary.gt.relationMap}
                            text={generateRecordSummary.text}
                            hoveredSpanId={hoveredSpanId}
                            setHoveredSpanId={setHoveredSpanId}
                        />

                        {/* Preds */}
                        <DocumentPanel
                            title="Model Predictions"
                            spans={generateRecordSummary.ml.spans}
                            relationMap={generateRecordSummary.ml.relationMap}
                            text={generateRecordSummary.text}
                            hoveredSpanId={hoveredSpanId}
                            setHoveredSpanId={setHoveredSpanId}
                        />

                        {/* Household Member Roster */}
                        <div className="panel">
                            <div className="panel-header">
                                <h2>Household Members</h2>
                                <span className="count-badge">{generateRecordSummary.household_members.length} members</span>
                            </div>
                            <div className="roster-view">
                                {generateRecordSummary.household_members.length === 0 ? (
                                    <p className="empty-state">No household members for this record.</p>
                                ) : (
                                    generateRecordSummary.household_members.map((member) => (
                                        <div
                                            key={member.id}
                                            className={`roster-card ${member.isResponsible ? 'primary-tenant' : ''}`}
                                        >
                                            <div className="roster-meta-line">
                                                <span className="text-bold">{member.fullName || 'Unknown Name'}</span>
                                                {member.isResponsible && <span className="badge-pill">Primary</span>}
                                            </div>
                                            <div className="roster-details">
                                                <span>ID: <code className="inline-code">{member.id}</code></span>
                                                <span>DOB: {member.dateOfBirth ? member.dateOfBirth.split(' ')[0] : 'N/A'}</span>
                                                <span>Tenure Type: {member.personTenureType || 'N/A'}</span>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Assignments table (Entity Linking) */}
                    <section className="matrix-panel">
                        <div className="matrix-header">
                            <h3>Needs & Assignments</h3>
                            <span className="count-badge">{generateRecordSummary.linking.length} rows</span>
                        </div>
                        <div className="table-wrapper">
                            <table className="matrix-table">
                                <thead>
                                    <tr>
                                        <th>Identified Need</th>
                                        <th>Need Type</th>
                                        <th>Assigned To</th>
                                        <th>Assignment Type</th>
                                        <th style={{ width: '10%' }}>Need Detection Conf.</th>
                                        <th style={{ width: '10%' }}>Relation Confidence</th>
                                        <th style={{ width: '10%' }}>Match Confidence</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {generateRecordSummary.linking.length === 0 ? (
                                        <tr>
                                            <td colSpan="7" className="empty-row">No Additional Needs found for this record.</td>
                                        </tr>
                                    ) : (
                                        generateRecordSummary.linking.map((link, index) => {
                                            const isHovered = hoveredSpanId === link.need_id;

                                            return (
                                                <tr
                                                    key={index}
                                                    className={isHovered ? "row-highlight" : ""}
                                                    onMouseEnter={() => setHoveredSpanId(link.need_id)}
                                                    onMouseLeave={() => setHoveredSpanId(null)}
                                                >
                                                    <td className="text-bold">
                                                        {link.need_text ? `"${link.need_text}"` : 'N/A'}
                                                    </td>
                                                    <td>
                                                        <span className="mono-label">{link.need_label || 'N/A'}</span>
                                                    </td>
                                                    <td className="text-bold">
                                                        {link.target_type === 'person' ? getPersonName(link.target_id) : link.target_id}
                                                    </td>
                                                    <td>
                                                        <span className={`type-tag tag-${link.target_type}`}>
                                                            {link.target_type || 'unassigned'}
                                                        </span>
                                                    </td>
                                                    <td className="mono-num">{link.need_conf ? `${(parseFloat(link.need_conf) * 100).toFixed(1)}%` : '-'}</td>
                                                    <td className="mono-num">{link.relation_conf ? `${(parseFloat(link.relation_conf) * 100).toFixed(1)}%` : '-'}</td>
                                                    <td className="mono-num">{link.linking_conf ? `${(parseFloat(link.linking_conf) * 100).toFixed(0)}%` : '-'}</td>
                                                </tr>
                                            );
                                        })
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </section>
                </main>
            )}
        </div>
    );
}