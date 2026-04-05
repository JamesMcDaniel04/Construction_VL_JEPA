import type { FormEvent } from "react";
import { startTransition, useEffect, useState } from "react";

type DashboardMetrics = {
  total_cases: number;
  analyzed_cases: number;
  escalated_cases: number;
  unresolved_cases: number;
  helpful_feedback_rate?: number | null;
  top_issue_classes: Array<{ issue_class: string; count: number }>;
};

type CaseSummary = {
  case_id: string;
  site_id: string;
  asset_id: string;
  status: string;
  top_issue_class?: string | null;
  escalation_recommendation?: string | null;
};

type CaseDetail = {
  case_id: string;
  site_id: string;
  asset_id: string;
  panel_family: string;
  status: string;
  latest_audit_id?: string | null;
  media_asset_id?: string | null;
  analysis?: {
    issue_candidates: Array<{ issue_class: string; confidence: number }>;
    next_steps: Array<{ step: string; citations: Array<{ title: string }> }>;
    escalation_recommendation: string;
    uncertainty_summary?: string | null;
    safety_notices: string[];
  } | null;
  feedback?: { labels: string[]; comment?: string | null } | null;
};

type AuditDetail = {
  audit: {
    audit_id: string;
    principal: string;
    created_at: string;
    outcome_status: string;
  };
  linked_assets: Array<{
    asset_id: string;
    asset_type: string;
    filename: string;
    presigned_url?: string | null;
    object_uri: string;
  }>;
};

type Identity = {
  display_name: string;
  role: string;
  organization_id?: string | null;
};

type Session = {
  apiBaseUrl: string;
  token: string;
};

const SESSION_KEY = "mtc.admin.session";

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [token, setToken] = useState("");
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [dashboard, setDashboard] = useState<DashboardMetrics | null>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseDetail | null>(null);
  const [selectedAudit, setSelectedAudit] = useState<AuditDetail | null>(null);
  const [documents, setDocuments] = useState<Array<{ document_id: string; title: string }>>([]);
  const [referenceStates, setReferenceStates] = useState<Array<{ state_id: string; state_label: string }>>([]);
  const [connecting, setConnecting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const [documentForm, setDocumentForm] = useState({
    documentId: "",
    sourceType: "manual",
    title: "",
    equipmentFamily: "electrical_panel_family_a",
    tags: "",
    file: null as File | null,
  });
  const [incidentForm, setIncidentForm] = useState({
    incidentId: "",
    title: "",
    summary: "",
    issueClass: "",
    fixSummary: "",
  });
  const [referenceForm, setReferenceForm] = useState({
    stateLabel: "",
    description: "",
    caption: "",
    equipmentFamily: "electrical_panel_family_a",
    file: null as File | null,
  });

  useEffect(() => {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) {
      return;
    }
    const session = JSON.parse(raw) as Session;
    setApiBaseUrl(session.apiBaseUrl);
    setToken(session.token);
    void connectWithCredentials(session.apiBaseUrl, session.token, true).catch((error) => {
      window.localStorage.removeItem(SESSION_KEY);
      setMessage(errorMessage(error));
    });
  }, []);

  useEffect(() => {
    if (identity) {
      window.localStorage.setItem(SESSION_KEY, JSON.stringify({ apiBaseUrl, token }));
    }
  }, [apiBaseUrl, identity, token]);

  async function connect() {
    setConnecting(true);
    setMessage(null);
    try {
      await connectWithCredentials(apiBaseUrl, token, false);
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setConnecting(false);
    }
  }

  async function connectWithCredentials(baseUrl: string, bearer: string, silent: boolean) {
    const me = await fetchJson<Identity>(`${baseUrl}/auth/me`, {
      headers: authHeaders(bearer),
    });
    setIdentity(me);
    await Promise.all([loadDashboard(baseUrl, bearer), loadCases(baseUrl, bearer), loadCorpus(baseUrl, bearer)]);
    if (!silent) {
      setMessage("Admin session connected.");
    }
  }

  async function loadDashboard(baseUrl = apiBaseUrl, bearer = token) {
    const payload = await fetchJson<DashboardMetrics>(`${baseUrl}/admin/dashboard`, {
      headers: authHeaders(bearer),
    });
    setDashboard(payload);
  }

  async function loadCases(baseUrl = apiBaseUrl, bearer = token) {
    const payload = await fetchJson<{ items: CaseSummary[] }>(`${baseUrl}/cases`, {
      headers: authHeaders(bearer),
    });
    startTransition(() => {
      setCases(payload.items);
    });
  }

  async function loadCorpus(baseUrl = apiBaseUrl, bearer = token) {
    const [docs, states] = await Promise.all([
      fetchJson<{ items: Array<{ document_id: string; title: string }> }>(
        `${baseUrl}/corpus/documents`,
        { headers: authHeaders(bearer) },
      ),
      fetchJson<{ items: Array<{ state_id: string; state_label: string }> }>(
        `${baseUrl}/reference-states`,
        { headers: authHeaders(bearer) },
      ),
    ]);
    setDocuments(docs.items);
    setReferenceStates(states.items);
  }

  async function openCase(caseId: string) {
    try {
      const detail = await fetchJson<CaseDetail>(`${apiBaseUrl}/cases/${caseId}`, {
        headers: authHeaders(token),
      });
      setSelectedCase(detail);
      if (detail.latest_audit_id) {
        const audit = await fetchJson<AuditDetail>(
          `${apiBaseUrl}/audit/triage/${detail.latest_audit_id}`,
          {
            headers: authHeaders(token),
          },
        );
        setSelectedAudit(audit);
      } else {
        setSelectedAudit(null);
      }
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function submitDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!documentForm.file) {
      setMessage("Select a manual, SOP, or OCR-compatible file first.");
      return;
    }
    const form = new FormData();
    form.append("file", documentForm.file);
    form.append("document_id", documentForm.documentId);
    form.append("source_type", documentForm.sourceType);
    form.append("equipment_family", documentForm.equipmentFamily);
    form.append("title", documentForm.title || documentForm.file.name);
    form.append("tags", documentForm.tags);
    try {
      await fetchJson(`${apiBaseUrl}/corpus/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });
      setDocumentForm((current) => ({ ...current, documentId: "", title: "", tags: "", file: null }));
      await loadCorpus();
      setMessage("Corpus document indexed.");
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function submitIncident(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      await fetchJson(`${apiBaseUrl}/corpus/incidents`, {
        method: "POST",
        headers: authHeaders(token),
        body: JSON.stringify({
          incident_id: incidentForm.incidentId,
          title: incidentForm.title,
          summary: incidentForm.summary,
          issue_class: incidentForm.issueClass,
          fix_summary: incidentForm.fixSummary,
        }),
      });
      setIncidentForm({
        incidentId: "",
        title: "",
        summary: "",
        issueClass: "",
        fixSummary: "",
      });
      setMessage("Incident indexed.");
      await loadDashboard();
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function submitReferenceState(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!referenceForm.file) {
      setMessage("Attach a reference-state image or short clip.");
      return;
    }
    const form = new FormData();
    form.append("file", referenceForm.file);
    form.append("state_label", referenceForm.stateLabel);
    form.append("description", referenceForm.description);
    form.append("caption", referenceForm.caption);
    form.append("equipment_family", referenceForm.equipmentFamily);
    try {
      await fetchJson(`${apiBaseUrl}/reference-states/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });
      setReferenceForm((current) => ({
        ...current,
        stateLabel: "",
        description: "",
        caption: "",
        file: null,
      }));
      await loadCorpus();
      setMessage("Reference state indexed.");
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  if (!identity) {
    return (
      <main className="shell auth-shell">
        <section className="hero-card">
          <p className="eyebrow">Pilot Admin</p>
          <h1>Electrical-panel field triage operations</h1>
          <p>
            Upload manuals, incidents, and reference states. Review technician cases, feedback,
            escalation rate, and unresolved work without touching raw API payloads.
          </p>
        </section>
        <section className="panel">
          <h2>Connect</h2>
          <label>
            API base URL
            <input value={apiBaseUrl} onChange={(event) => setApiBaseUrl(event.target.value)} />
          </label>
          <label>
            Admin bearer token
            <input
              value={token}
              onChange={(event) => setToken(event.target.value)}
              type="password"
            />
          </label>
          <button className="primary-button" onClick={() => void connect()} disabled={connecting}>
            {connecting ? "Connecting…" : "Connect to admin console"}
          </button>
          {message ? <p className="inline-message">{message}</p> : null}
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Supervisor Console</p>
          <h1>{identity.display_name}</h1>
          <p className="subtle">
            {identity.role} · {identity.organization_id ?? "service"} · one electrical-panel family
          </p>
        </div>
        <div className="topbar-actions">
          <button className="ghost-button" onClick={() => void Promise.all([loadDashboard(), loadCases(), loadCorpus()])}>
            Refresh
          </button>
          <button
            className="ghost-button"
            onClick={() => {
              window.localStorage.removeItem(SESSION_KEY);
              setIdentity(null);
              setToken("");
              setSelectedAudit(null);
              setSelectedCase(null);
            }}
          >
            Sign out
          </button>
        </div>
      </header>

      {message ? <div className="message-banner">{message}</div> : null}

      <section className="metrics-grid">
        <MetricCard label="Total cases" value={String(dashboard?.total_cases ?? 0)} />
        <MetricCard label="Analyzed" value={String(dashboard?.analyzed_cases ?? 0)} />
        <MetricCard label="Escalated" value={String(dashboard?.escalated_cases ?? 0)} />
        <MetricCard label="Unresolved" value={String(dashboard?.unresolved_cases ?? 0)} />
      </section>

      {dashboard?.top_issue_classes?.length ? (
        <section className="panel">
          <h2>Current pilot signals</h2>
          <div className="signal-grid">
            <div>
              <p className="subtle">Top issue classes</p>
              <ul>
                {dashboard.top_issue_classes.map((item) => (
                  <li key={item.issue_class}>
                    {item.issue_class} · {item.count}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="subtle">Helpful feedback rate</p>
              <strong className="rate-pill">
                {dashboard.helpful_feedback_rate == null
                  ? "No feedback yet"
                  : `${Math.round(dashboard.helpful_feedback_rate * 100)}%`}
              </strong>
            </div>
          </div>
        </section>
      ) : null}

      <section className="two-column">
        <div className="panel">
          <h2>Upload manual / SOP</h2>
          <form className="stack" onSubmit={submitDocument}>
            <input
              placeholder="document_id"
              value={documentForm.documentId}
              onChange={(event) => setDocumentForm({ ...documentForm, documentId: event.target.value })}
              required
            />
            <input
              placeholder="title"
              value={documentForm.title}
              onChange={(event) => setDocumentForm({ ...documentForm, title: event.target.value })}
            />
            <input
              placeholder="comma-separated tags"
              value={documentForm.tags}
              onChange={(event) => setDocumentForm({ ...documentForm, tags: event.target.value })}
            />
            <select
              value={documentForm.sourceType}
              onChange={(event) => setDocumentForm({ ...documentForm, sourceType: event.target.value })}
            >
              <option value="manual">manual</option>
              <option value="sop">sop</option>
              <option value="ticket">ticket</option>
              <option value="repair_note">repair_note</option>
            </select>
            <input
              placeholder="equipment family"
              value={documentForm.equipmentFamily}
              onChange={(event) =>
                setDocumentForm({ ...documentForm, equipmentFamily: event.target.value })
              }
            />
            <input
              type="file"
              onChange={(event) =>
                setDocumentForm({ ...documentForm, file: event.target.files?.[0] ?? null })
              }
              required
            />
            <button className="primary-button" type="submit">
              Index document
            </button>
          </form>
          <div className="compact-list">
            {documents.slice(0, 6).map((document) => (
              <div key={document.document_id} className="list-row">
                <strong>{document.title}</strong>
                <span>{document.document_id}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Add incident</h2>
          <form className="stack" onSubmit={submitIncident}>
            <input
              placeholder="incident_id"
              value={incidentForm.incidentId}
              onChange={(event) => setIncidentForm({ ...incidentForm, incidentId: event.target.value })}
              required
            />
            <input
              placeholder="title"
              value={incidentForm.title}
              onChange={(event) => setIncidentForm({ ...incidentForm, title: event.target.value })}
              required
            />
            <textarea
              placeholder="summary"
              value={incidentForm.summary}
              onChange={(event) => setIncidentForm({ ...incidentForm, summary: event.target.value })}
              required
            />
            <input
              placeholder="issue class"
              value={incidentForm.issueClass}
              onChange={(event) => setIncidentForm({ ...incidentForm, issueClass: event.target.value })}
              required
            />
            <textarea
              placeholder="fix summary"
              value={incidentForm.fixSummary}
              onChange={(event) => setIncidentForm({ ...incidentForm, fixSummary: event.target.value })}
              required
            />
            <button className="primary-button" type="submit">
              Index incident
            </button>
          </form>
        </div>
      </section>

      <section className="two-column">
        <div className="panel">
          <h2>Reference-state library</h2>
          <form className="stack" onSubmit={submitReferenceState}>
            <input
              placeholder="state label"
              value={referenceForm.stateLabel}
              onChange={(event) => setReferenceForm({ ...referenceForm, stateLabel: event.target.value })}
              required
            />
            <textarea
              placeholder="description"
              value={referenceForm.description}
              onChange={(event) =>
                setReferenceForm({ ...referenceForm, description: event.target.value })
              }
              required
            />
            <textarea
              placeholder="caption"
              value={referenceForm.caption}
              onChange={(event) => setReferenceForm({ ...referenceForm, caption: event.target.value })}
              required
            />
            <input
              type="file"
              onChange={(event) =>
                setReferenceForm({ ...referenceForm, file: event.target.files?.[0] ?? null })
              }
              required
            />
            <button className="primary-button" type="submit">
              Add reference state
            </button>
          </form>
          <div className="compact-list">
            {referenceStates.slice(0, 6).map((state) => (
              <div key={state.state_id} className="list-row">
                <strong>{state.state_label}</strong>
                <span>{state.state_id}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Case review</h2>
          <div className="compact-list">
            {cases.map((item) => (
              <button
                key={item.case_id}
                className="case-button"
                onClick={() => void openCase(item.case_id)}
              >
                <div>
                  <strong>
                    {item.site_id} · {item.asset_id}
                  </strong>
                  <p>{item.top_issue_class ?? "awaiting analysis"}</p>
                </div>
                <span>{item.status}</span>
              </button>
            ))}
          </div>
          {selectedCase ? (
            <article className="case-detail">
              <h3>{selectedCase.case_id}</h3>
              <p className="subtle">
                {selectedCase.site_id} · {selectedCase.asset_id} · {selectedCase.status}
              </p>
              {selectedCase.analysis ? (
                <>
                  <p>{selectedCase.analysis.uncertainty_summary}</p>
                  <ul>
                    {selectedCase.analysis.issue_candidates.map((item) => (
                      <li key={item.issue_class}>
                        {item.issue_class} · {Math.round(item.confidence * 100)}%
                      </li>
                    ))}
                  </ul>
                  <h4>Next steps</h4>
                  <ul>
                    {selectedCase.analysis.next_steps.map((step) => (
                      <li key={step.step}>
                        {step.step} · {step.citations[0]?.title ?? "No citation"}
                      </li>
                    ))}
                  </ul>
                  <h4>Safety notices</h4>
                  <ul>
                    {selectedCase.analysis.safety_notices.map((notice) => (
                      <li key={notice}>{notice}</li>
                    ))}
                  </ul>
                </>
              ) : (
                <p className="subtle">No analysis yet.</p>
              )}
              {selectedCase.feedback ? (
                <p className="subtle">
                  Feedback: {selectedCase.feedback.labels.join(", ")}
                  {selectedCase.feedback.comment ? ` · ${selectedCase.feedback.comment}` : ""}
                </p>
              ) : null}
              {selectedAudit ? (
                <>
                  <h4>Linked audit evidence</h4>
                  <p className="subtle">
                    {selectedAudit.audit.audit_id} · {selectedAudit.audit.principal} ·{" "}
                    {selectedAudit.audit.outcome_status}
                  </p>
                  <ul>
                    {selectedAudit.linked_assets.map((asset) => (
                      <li key={asset.asset_id}>
                        <a
                          href={asset.presigned_url ?? asset.object_uri}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {asset.filename}
                        </a>{" "}
                        · {asset.asset_type}
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
            </article>
          ) : null}
        </div>
      </section>
    </main>
  );
}

function MetricCard(props: { label: string; value: string }) {
  return (
    <article className="metric-card">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </article>
  );
}

function authHeaders(token: string) {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw payload;
  }
  return payload as T;
}

function errorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (typeof detail === "object" && detail !== null && "message" in detail) {
      return String((detail as { message: unknown }).message);
    }
  }
  return "The request could not be completed.";
}
