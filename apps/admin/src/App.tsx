import type { FormEvent } from "react";
import { createClient, type Session } from "@supabase/supabase-js";
import { startTransition, useEffect, useMemo, useState } from "react";
import { helpfulRateLabel, inviteIssuedMessage } from "./lib/pilotAdmin";

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
  email?: string | null;
};

type PilotUserView = {
  user_id: string;
  organization_id: string;
  role: string;
  display_name: string;
  email?: string | null;
  created_at: string;
  active: boolean;
};

type SiteRecord = {
  site_id: string;
  organization_id: string;
  name: string;
  code?: string | null;
  active: boolean;
};

type AssetRecord = {
  asset_id: string;
  organization_id: string;
  site_id: string;
  display_name: string;
  panel_family: string;
  equipment_family: string;
  panel_id?: string | null;
  active: boolean;
};

const DEV_API_BASE_KEY = "mtc.admin.dev-api-base";
const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_MTC_API_BASE_URL ?? "http://localhost:8000";
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL ?? "";
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY ?? "";
const SUPABASE_REDIRECT_URL =
  import.meta.env.VITE_SUPABASE_WEB_REDIRECT_URL ??
  `${window.location.origin}/auth/callback`;
const DEBUG_OVERRIDE_ENABLED =
  import.meta.env.DEV || import.meta.env.VITE_MTC_ENABLE_DEBUG_OVERRIDE === "1";

const supabase =
  SUPABASE_URL && SUPABASE_ANON_KEY
    ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
        },
      })
    : null;

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [debugApiBaseUrl, setDebugApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [dashboard, setDashboard] = useState<DashboardMetrics | null>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseDetail | null>(null);
  const [selectedAudit, setSelectedAudit] = useState<AuditDetail | null>(null);
  const [documents, setDocuments] = useState<Array<{ document_id: string; title: string }>>([]);
  const [referenceStates, setReferenceStates] = useState<
    Array<{ state_id: string; state_label: string }>
  >([]);
  const [pilotUsers, setPilotUsers] = useState<PilotUserView[]>([]);
  const [sites, setSites] = useState<SiteRecord[]>([]);
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [connecting, setConnecting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [loginEmail, setLoginEmail] = useState("");
  const [developerTapCount, setDeveloperTapCount] = useState(0);
  const [debugVisible, setDebugVisible] = useState(false);

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
  const [inviteForm, setInviteForm] = useState({
    displayName: "",
    email: "",
    role: "technician",
  });
  const [siteForm, setSiteForm] = useState({
    siteId: "",
    name: "",
    code: "",
  });
  const [assetForm, setAssetForm] = useState({
    assetId: "",
    siteId: "",
    displayName: "",
    panelFamily: "",
    equipmentFamily: "electrical_panel_family_a",
    panelId: "",
  });

  const sitesById = useMemo(() => new Map(sites.map((site) => [site.site_id, site])), [sites]);

  useEffect(() => {
    void loadBootstrapState();
  }, []);

  useEffect(() => {
    if (!supabase) {
      return undefined;
    }
    const subscription = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      if (nextSession) {
        void connectWithSession(nextSession, true);
      } else {
        setIdentity(null);
        setDashboard(null);
        setCases([]);
        setSelectedCase(null);
        setSelectedAudit(null);
      }
    });
    return () => {
      subscription.data.subscription.unsubscribe();
    };
  }, [apiBaseUrl]);

  async function loadBootstrapState() {
    try {
      const [debugBaseUrl, sessionResponse] = await Promise.all([
        DEBUG_OVERRIDE_ENABLED
          ? Promise.resolve(window.localStorage.getItem(DEV_API_BASE_KEY))
          : Promise.resolve(null),
        supabase?.auth.getSession() ?? Promise.resolve({ data: { session: null } }),
      ]);
      const resolvedBaseUrl = debugBaseUrl?.trim() ? debugBaseUrl.trim() : DEFAULT_API_BASE_URL;
      setApiBaseUrl(resolvedBaseUrl);
      setDebugApiBaseUrl(resolvedBaseUrl);

      const restoredSession = sessionResponse.data.session;
      setSession(restoredSession);
      if (restoredSession) {
        await connectWithSession(restoredSession, true);
      }
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function sendMagicLink() {
    if (!supabase) {
      setMessage(
        "Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY for the admin build.",
      );
      return;
    }
    if (!loginEmail.trim()) {
      setMessage("Enter the invited admin email for this organization.");
      return;
    }
    setConnecting(true);
    setMessage(null);
    try {
      const { error } = await supabase.auth.signInWithOtp({
        email: loginEmail.trim().toLowerCase(),
        options: { emailRedirectTo: SUPABASE_REDIRECT_URL },
      });
      if (error) {
        throw error;
      }
      setMessage("Magic link sent. Open it in this browser to continue.");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setConnecting(false);
    }
  }

  async function connectWithSession(nextSession: Session, silent: boolean) {
    const accessToken = nextSession.access_token;
    const me = await fetchJson<Identity>(`${apiBaseUrl}/auth/me`, {
      headers: authHeaders(accessToken),
    });
    setIdentity(me);
    await Promise.all([
      loadDashboard(accessToken),
      loadCases(accessToken),
      loadCorpus(accessToken),
      loadPilotUsers(accessToken),
      loadCatalog(accessToken),
    ]);
    if (!silent) {
      setMessage("Admin session connected.");
    }
  }

  async function loadDashboard(accessToken = session?.access_token) {
    if (!accessToken) {
      return;
    }
    const payload = await fetchJson<DashboardMetrics>(`${apiBaseUrl}/admin/dashboard`, {
      headers: authHeaders(accessToken),
    });
    setDashboard(payload);
  }

  async function loadCases(accessToken = session?.access_token) {
    if (!accessToken) {
      return;
    }
    const payload = await fetchJson<{ items: CaseSummary[] }>(`${apiBaseUrl}/cases`, {
      headers: authHeaders(accessToken),
    });
    startTransition(() => {
      setCases(payload.items);
    });
  }

  async function loadCorpus(accessToken = session?.access_token) {
    if (!accessToken) {
      return;
    }
    const [docs, states] = await Promise.all([
      fetchJson<{ items: Array<{ document_id: string; title: string }> }>(
        `${apiBaseUrl}/corpus/documents`,
        { headers: authHeaders(accessToken) },
      ),
      fetchJson<{ items: Array<{ state_id: string; state_label: string }> }>(
        `${apiBaseUrl}/reference-states`,
        { headers: authHeaders(accessToken) },
      ),
    ]);
    setDocuments(docs.items);
    setReferenceStates(states.items);
  }

  async function loadPilotUsers(accessToken = session?.access_token) {
    if (!accessToken) {
      return;
    }
    const payload = await fetchJson<{ items: PilotUserView[] }>(`${apiBaseUrl}/admin/pilot-users`, {
      headers: authHeaders(accessToken),
    });
    setPilotUsers(payload.items);
  }

  async function loadCatalog(accessToken = session?.access_token) {
    if (!accessToken) {
      return;
    }
    const [sitePayload, assetPayload] = await Promise.all([
      fetchJson<{ items: SiteRecord[] }>(`${apiBaseUrl}/catalog/sites`, {
        headers: authHeaders(accessToken),
      }),
      fetchJson<{ items: AssetRecord[] }>(`${apiBaseUrl}/catalog/assets`, {
        headers: authHeaders(accessToken),
      }),
    ]);
    setSites(sitePayload.items);
    setAssets(assetPayload.items);
  }

  async function openCase(caseId: string) {
    if (!session) {
      return;
    }
    try {
      const detail = await fetchJson<CaseDetail>(`${apiBaseUrl}/cases/${caseId}`, {
        headers: authHeaders(session.access_token),
      });
      setSelectedCase(detail);
      if (detail.latest_audit_id) {
        const audit = await fetchJson<AuditDetail>(
          `${apiBaseUrl}/audit/triage/${detail.latest_audit_id}`,
          {
            headers: authHeaders(session.access_token),
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
    if (!session) {
      return;
    }
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
        headers: { Authorization: `Bearer ${session.access_token}` },
        body: form,
      });
      setDocumentForm((current) => ({
        ...current,
        documentId: "",
        title: "",
        tags: "",
        file: null,
      }));
      await loadCorpus();
      setMessage("Corpus document indexed.");
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function submitIncident(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    try {
      await fetchJson(`${apiBaseUrl}/corpus/incidents`, {
        method: "POST",
        headers: authHeaders(session.access_token),
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
    if (!session) {
      return;
    }
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
        headers: { Authorization: `Bearer ${session.access_token}` },
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

  async function invitePilotUser(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session || !identity) {
      return;
    }
    try {
      const payload = await fetchJson<{
        user: PilotUserView;
        invite_status: string;
      }>(`${apiBaseUrl}/admin/pilot-users/invite`, {
        method: "POST",
        headers: authHeaders(session.access_token),
        body: JSON.stringify({
          organization_id: identity.organization_id ?? "org-1",
          role: inviteForm.role,
          display_name: inviteForm.displayName,
          email: inviteForm.email,
        }),
      });
      setInviteForm({ displayName: "", email: "", role: "technician" });
      await loadPilotUsers();
      setMessage(
        inviteIssuedMessage(payload.user.display_name, payload.user.email ?? "", payload.invite_status),
      );
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function submitSite(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    try {
      await fetchJson(`${apiBaseUrl}/catalog/sites`, {
        method: "POST",
        headers: authHeaders(session.access_token),
        body: JSON.stringify({
          site_id: siteForm.siteId,
          name: siteForm.name,
          code: emptyToUndefined(siteForm.code),
        }),
      });
      setSiteForm({ siteId: "", name: "", code: "" });
      await loadCatalog();
      setMessage("Site saved.");
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function submitAsset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session) {
      return;
    }
    try {
      await fetchJson(`${apiBaseUrl}/catalog/assets`, {
        method: "POST",
        headers: authHeaders(session.access_token),
        body: JSON.stringify({
          asset_id: assetForm.assetId,
          site_id: assetForm.siteId,
          display_name: assetForm.displayName,
          panel_family: assetForm.panelFamily,
          equipment_family: assetForm.equipmentFamily,
          panel_id: emptyToUndefined(assetForm.panelId),
        }),
      });
      setAssetForm({
        assetId: "",
        siteId: "",
        displayName: "",
        panelFamily: "",
        equipmentFamily: "electrical_panel_family_a",
        panelId: "",
      });
      await loadCatalog();
      setMessage("Asset saved.");
    } catch (error) {
      setMessage(errorMessage(error));
    }
  }

  async function saveDebugBaseUrl() {
    const next = debugApiBaseUrl.trim() || DEFAULT_API_BASE_URL;
    setApiBaseUrl(next);
    if (DEBUG_OVERRIDE_ENABLED) {
      window.localStorage.setItem(DEV_API_BASE_KEY, next);
    }
    setMessage(`Developer API override set to ${next}.`);
    setDebugVisible(false);
    if (session) {
      await connectWithSession(session, true);
    }
  }

  function resetDebugBaseUrl() {
    setApiBaseUrl(DEFAULT_API_BASE_URL);
    setDebugApiBaseUrl(DEFAULT_API_BASE_URL);
    window.localStorage.removeItem(DEV_API_BASE_KEY);
    setMessage("Developer API override cleared.");
    setDebugVisible(false);
  }

  function onBuildLabelClick() {
    if (!DEBUG_OVERRIDE_ENABLED) {
      return;
    }
    setDeveloperTapCount((current) => {
      const next = current + 1;
      if (next >= 5) {
        setDebugVisible(true);
        return 0;
      }
      return next;
    });
  }

  async function signOut() {
    if (supabase) {
      await supabase.auth.signOut();
    }
    setIdentity(null);
    setSession(null);
    setDashboard(null);
    setCases([]);
    setSelectedAudit(null);
    setSelectedCase(null);
    setSites([]);
    setAssets([]);
    setMessage("Signed out.");
  }

  if (!identity) {
    return (
      <main className="shell auth-shell">
        <section className="hero-card">
          <p className="eyebrow">Pilot Admin</p>
          <h1>Electrical-panel field triage operations</h1>
          <p>
            Sign in with your invited email, then manage corpus uploads, reference states,
            pilot users, sites, assets, and technician case review without touching raw tokens.
          </p>
        </section>
        <section className="panel">
          <h2>Sign in with magic link</h2>
          <label>
            Invited email
            <input
              value={loginEmail}
              onChange={(event) => setLoginEmail(event.target.value)}
              type="email"
              placeholder="admin@example.com"
            />
          </label>
          <button className="primary-button" onClick={() => void sendMagicLink()} disabled={connecting}>
            {connecting ? "Sending…" : "Send magic link"}
          </button>
          {message ? <p className="inline-message">{message}</p> : null}
        </section>
        <button className="debug-trigger" onClick={onBuildLabelClick}>
          pilot build · click 5x for developer override
        </button>
        {debugVisible ? (
          <section className="panel">
            <h2>Developer override</h2>
            <label>
              API base URL
              <input
                value={debugApiBaseUrl}
                onChange={(event) => setDebugApiBaseUrl(event.target.value)}
              />
            </label>
            <div className="topbar-actions">
              <button className="primary-button" onClick={() => void saveDebugBaseUrl()}>
                Save override
              </button>
              <button className="ghost-button" onClick={resetDebugBaseUrl}>
                Reset
              </button>
            </div>
          </section>
        ) : null}
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
          <button
            className="ghost-button"
            onClick={() =>
              void Promise.all([loadDashboard(), loadCases(), loadCorpus(), loadPilotUsers(), loadCatalog()])
            }
          >
            Refresh
          </button>
          <button className="ghost-button" onClick={() => void signOut()}>
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
                {helpfulRateLabel(dashboard.helpful_feedback_rate)}
              </strong>
            </div>
          </div>
        </section>
      ) : null}

      <section className="three-column">
        <div className="panel">
          <h2>Pilot user invites</h2>
          <form className="stack" onSubmit={invitePilotUser}>
            <input
              placeholder="display name"
              value={inviteForm.displayName}
              onChange={(event) => setInviteForm({ ...inviteForm, displayName: event.target.value })}
              required
            />
            <input
              placeholder="email"
              value={inviteForm.email}
              onChange={(event) => setInviteForm({ ...inviteForm, email: event.target.value })}
              type="email"
              required
            />
            <select
              value={inviteForm.role}
              onChange={(event) => setInviteForm({ ...inviteForm, role: event.target.value })}
            >
              <option value="technician">technician</option>
              <option value="admin">admin</option>
            </select>
            <button className="primary-button" type="submit">
              Send invite
            </button>
          </form>
          <div className="compact-list">
            {pilotUsers.slice(0, 8).map((user) => (
              <div key={user.user_id} className="list-row">
                <div>
                  <strong>{user.display_name}</strong>
                  <p className="subtle">
                    {user.role} · {user.email ?? "no email"}
                  </p>
                </div>
                <span>{user.user_id}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Sites</h2>
          <form className="stack" onSubmit={submitSite}>
            <input
              placeholder="site_id"
              value={siteForm.siteId}
              onChange={(event) => setSiteForm({ ...siteForm, siteId: event.target.value })}
              required
            />
            <input
              placeholder="name"
              value={siteForm.name}
              onChange={(event) => setSiteForm({ ...siteForm, name: event.target.value })}
              required
            />
            <input
              placeholder="code"
              value={siteForm.code}
              onChange={(event) => setSiteForm({ ...siteForm, code: event.target.value })}
            />
            <button className="primary-button" type="submit">
              Save site
            </button>
          </form>
          <div className="compact-list">
            {sites.slice(0, 8).map((site) => (
              <div key={site.site_id} className="list-row">
                <div>
                  <strong>{site.name}</strong>
                  <p className="subtle">{site.code ?? "no code"}</p>
                </div>
                <span>{site.site_id}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Assets</h2>
          <form className="stack" onSubmit={submitAsset}>
            <input
              placeholder="asset_id"
              value={assetForm.assetId}
              onChange={(event) => setAssetForm({ ...assetForm, assetId: event.target.value })}
              required
            />
            <select
              value={assetForm.siteId}
              onChange={(event) => setAssetForm({ ...assetForm, siteId: event.target.value })}
              required
            >
              <option value="">select site</option>
              {sites.map((site) => (
                <option key={site.site_id} value={site.site_id}>
                  {site.name} · {site.site_id}
                </option>
              ))}
            </select>
            <input
              placeholder="display name"
              value={assetForm.displayName}
              onChange={(event) =>
                setAssetForm({ ...assetForm, displayName: event.target.value })
              }
              required
            />
            <input
              placeholder="panel family"
              value={assetForm.panelFamily}
              onChange={(event) =>
                setAssetForm({ ...assetForm, panelFamily: event.target.value })
              }
              required
            />
            <input
              placeholder="equipment family"
              value={assetForm.equipmentFamily}
              onChange={(event) =>
                setAssetForm({ ...assetForm, equipmentFamily: event.target.value })
              }
              required
            />
            <input
              placeholder="panel id"
              value={assetForm.panelId}
              onChange={(event) => setAssetForm({ ...assetForm, panelId: event.target.value })}
            />
            <button className="primary-button" type="submit">
              Save asset
            </button>
          </form>
          <div className="compact-list">
            {assets.slice(0, 8).map((asset) => (
              <div key={asset.asset_id} className="list-row">
                <div>
                  <strong>{asset.display_name}</strong>
                  <p className="subtle">
                    {sitesById.get(asset.site_id)?.name ?? asset.site_id} · {asset.panel_family}
                  </p>
                </div>
                <span>{asset.asset_id}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="two-column">
        <div className="panel">
          <h2>Upload manual / SOP</h2>
          <form className="stack" onSubmit={submitDocument}>
            <input
              placeholder="document_id"
              value={documentForm.documentId}
              onChange={(event) =>
                setDocumentForm({ ...documentForm, documentId: event.target.value })
              }
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
              onChange={(event) =>
                setDocumentForm({ ...documentForm, sourceType: event.target.value })
              }
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
              onChange={(event) =>
                setIncidentForm({ ...incidentForm, issueClass: event.target.value })
              }
              required
            />
            <textarea
              placeholder="fix summary"
              value={incidentForm.fixSummary}
              onChange={(event) =>
                setIncidentForm({ ...incidentForm, fixSummary: event.target.value })
              }
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
              onChange={(event) =>
                setReferenceForm({ ...referenceForm, stateLabel: event.target.value })
              }
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
              placeholder="equipment family"
              value={referenceForm.equipmentFamily}
              onChange={(event) =>
                setReferenceForm({ ...referenceForm, equipmentFamily: event.target.value })
              }
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

      <button className="debug-trigger" onClick={onBuildLabelClick}>
        pilot build · click 5x for developer override
      </button>
      {debugVisible ? (
        <section className="panel">
          <h2>Developer override</h2>
          <label>
            API base URL
            <input
              value={debugApiBaseUrl}
              onChange={(event) => setDebugApiBaseUrl(event.target.value)}
            />
          </label>
          <div className="topbar-actions">
            <button className="primary-button" onClick={() => void saveDebugBaseUrl()}>
              Save override
            </button>
            <button className="ghost-button" onClick={resetDebugBaseUrl}>
              Reset
            </button>
          </div>
        </section>
      ) : null}
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

function authHeaders(accessToken: string) {
  return {
    Authorization: `Bearer ${accessToken}`,
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

function emptyToUndefined(value: string): string | undefined {
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
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
