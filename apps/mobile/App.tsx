import "react-native-url-polyfill/auto";

import type { Dispatch, ReactNode, SetStateAction } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { createClient, type Session } from "@supabase/supabase-js";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { startTransition, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import {
  pendingCaseLabel,
  pendingCaseStatusMessage,
  restoreDraftCase,
} from "./src/lib/pilotState";

type Identity = {
  principal: string;
  principal_type: string;
  role: string;
  user_id?: string | null;
  organization_id?: string | null;
  display_name: string;
  email?: string | null;
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

type CaptureAsset = {
  uri: string;
  name: string;
  mimeType: string;
  mediaType: "image" | "video";
};

type DraftCase = {
  siteId: string;
  assetId: string;
  question: string;
  operatorContext: string;
  expectedStateLabel: string;
  pendingCaseId?: string | null;
  selectedMedia?: CaptureAsset | null;
};

type CaseSummary = {
  case_id: string;
  site_id: string;
  asset_id: string;
  panel_family: string;
  panel_id?: string | null;
  status: string;
  top_issue_class?: string | null;
  escalation_recommendation?: string | null;
  helpful?: boolean | null;
};

type TriageResponse = {
  issue_candidates: Array<{ issue_class: string; confidence: number; rationale: string }>;
  state_assessment: {
    matched_state_label?: string | null;
    summary: string;
    confidence: number;
    matches_expected?: boolean | null;
  };
  next_steps: Array<{
    step: string;
    confidence: number;
    citations: Array<{ title: string; snippet?: string | null }>;
  }>;
  similar_incidents: Array<{
    title: string;
    issue_class: string;
    fix_summary: string;
    similarity: number;
  }>;
  escalation_recommendation: string;
  visual_evidence_status: string;
  uncertainty_summary?: string | null;
  safety_notices: string[];
};

type CaseDetail = {
  case_id: string;
  site_id: string;
  asset_id: string;
  panel_family: string;
  panel_id?: string | null;
  status: string;
  analysis?: TriageResponse | null;
  feedback?: { labels: string[]; comment?: string | null } | null;
};

const DRAFT_KEY = "mtc.mobile.draft";
const RECENT_ASSET_IDS_KEY = "mtc.mobile.recent-assets";
const DEV_API_BASE_KEY = "mtc.mobile.dev-api-base";
const DEFAULT_API_BASE_URL =
  process.env.EXPO_PUBLIC_MTC_API_BASE_URL ?? "http://localhost:8000";
const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL ?? "";
const SUPABASE_ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY ?? "";
const MOBILE_REDIRECT_URL =
  process.env.EXPO_PUBLIC_SUPABASE_MOBILE_REDIRECT_URL ?? "mtc://auth/callback";
const DEBUG_OVERRIDE_ENABLED =
  __DEV__ || process.env.EXPO_PUBLIC_MTC_ENABLE_DEBUG_OVERRIDE === "1";
const DEFAULT_DRAFT: DraftCase = {
  siteId: "",
  assetId: "",
  question: "",
  operatorContext: "",
  expectedStateLabel: "",
  pendingCaseId: null,
  selectedMedia: null,
};

const supabase =
  SUPABASE_URL && SUPABASE_ANON_KEY
    ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: {
          storage: AsyncStorage as never,
          autoRefreshToken: true,
          persistSession: true,
          detectSessionInUrl: false,
        },
      })
    : null;

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [draft, setDraft] = useState<DraftCase>(DEFAULT_DRAFT);
  const [recentCases, setRecentCases] = useState<CaseSummary[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseDetail | null>(null);
  const [sites, setSites] = useState<SiteRecord[]>([]);
  const [assets, setAssets] = useState<AssetRecord[]>([]);
  const [recentAssetIds, setRecentAssetIds] = useState<string[]>([]);
  const [loadingSession, setLoadingSession] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [loadingCatalog, setLoadingCatalog] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);
  const [captureHints, setCaptureHints] = useState<string[]>([]);
  const [authEmail, setAuthEmail] = useState("");
  const [loginMessage, setLoginMessage] = useState<string | null>(null);
  const [siteQuery, setSiteQuery] = useState("");
  const [assetQuery, setAssetQuery] = useState("");
  const [developerTapCount, setDeveloperTapCount] = useState(0);
  const [debugVisible, setDebugVisible] = useState(false);
  const [debugApiBaseUrl, setDebugApiBaseUrl] = useState(DEFAULT_API_BASE_URL);

  const selectedSite = useMemo(
    () => sites.find((item) => item.site_id === draft.siteId) ?? null,
    [draft.siteId, sites],
  );
  const selectedAsset = useMemo(
    () => assets.find((item) => item.asset_id === draft.assetId) ?? null,
    [assets, draft.assetId],
  );
  const filteredSites = useMemo(() => {
    const query = siteQuery.trim().toLowerCase();
    if (!query) {
      return sites;
    }
    return sites.filter((site) => {
      return (
        site.name.toLowerCase().includes(query) ||
        site.site_id.toLowerCase().includes(query) ||
        (site.code ?? "").toLowerCase().includes(query)
      );
    });
  }, [siteQuery, sites]);
  const recentAssets = useMemo(() => {
    const byId = new Map(assets.map((asset) => [asset.asset_id, asset]));
    return recentAssetIds
      .map((assetId) => byId.get(assetId))
      .filter((asset): asset is AssetRecord => asset != null);
  }, [assets, recentAssetIds]);
  const pendingLabel = useMemo(() => pendingCaseLabel(draft.pendingCaseId), [draft.pendingCaseId]);

  useEffect(() => {
    void loadBootstrapState();
  }, []);

  useEffect(() => {
    if (!loadingSession) {
      void AsyncStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
      void AsyncStorage.setItem(RECENT_ASSET_IDS_KEY, JSON.stringify(recentAssetIds));
    }
  }, [draft, loadingSession, recentAssetIds]);

  useEffect(() => {
    if (!DEBUG_OVERRIDE_ENABLED) {
      return;
    }
    void AsyncStorage.setItem(DEV_API_BASE_KEY, debugApiBaseUrl);
  }, [debugApiBaseUrl]);

  useEffect(() => {
    if (!supabase) {
      return undefined;
    }

    const subscription = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      if (!loadingSession) {
        if (nextSession) {
          void connectWithSession(nextSession, {
            silent: true,
            pendingCaseId: draft.pendingCaseId ?? null,
          });
        } else {
          setIdentity(null);
          setRecentCases([]);
          setSelectedCase(null);
          setSites([]);
          setAssets([]);
        }
      }
    });

    void Linking.getInitialURL().then((url) => {
      if (url) {
        void handleIncomingUrl(url);
      }
    });
    const linkSubscription = Linking.addEventListener("url", ({ url }) => {
      void handleIncomingUrl(url);
    });

    return () => {
      subscription.data.subscription.unsubscribe();
      linkSubscription.remove();
    };
  }, [apiBaseUrl, draft.pendingCaseId, loadingSession]);

  useEffect(() => {
    if (!identity || !draft.siteId) {
      setAssets([]);
      return;
    }
    void loadAssets();
  }, [apiBaseUrl, assetQuery, draft.siteId, identity, session]);

  async function loadBootstrapState() {
    try {
      const [draftRaw, recentRaw, debugBaseUrl, sessionResponse] = await Promise.all([
        AsyncStorage.getItem(DRAFT_KEY),
        AsyncStorage.getItem(RECENT_ASSET_IDS_KEY),
        DEBUG_OVERRIDE_ENABLED ? AsyncStorage.getItem(DEV_API_BASE_KEY) : Promise.resolve(null),
        supabase?.auth.getSession() ?? Promise.resolve({ data: { session: null } }),
      ]);

      const restoredDraft = restoreDraftCase(DEFAULT_DRAFT, draftRaw);
      setDraft(restoredDraft);

      if (recentRaw) {
        try {
          const parsed = JSON.parse(recentRaw) as string[];
          setRecentAssetIds(Array.isArray(parsed) ? parsed : []);
        } catch {
          setRecentAssetIds([]);
        }
      }

      const resolvedBaseUrl = debugBaseUrl?.trim() ? debugBaseUrl.trim() : DEFAULT_API_BASE_URL;
      setApiBaseUrl(resolvedBaseUrl);
      setDebugApiBaseUrl(resolvedBaseUrl);

      const restoredSession = sessionResponse.data.session;
      setSession(restoredSession);
      if (restoredSession) {
        await connectWithSession(restoredSession, {
          silent: true,
          pendingCaseId: restoredDraft.pendingCaseId ?? null,
        });
      }
    } catch (error) {
      setSyncMessage(errorMessage(error));
      setIdentity(null);
      setSession(null);
    } finally {
      setLoadingSession(false);
    }
  }

  async function handleIncomingUrl(url: string) {
    if (!supabase) {
      return;
    }
    const params = extractUrlParams(url);
    const accessToken = params.get("access_token");
    const refreshToken = params.get("refresh_token");
    const code = params.get("code");
    if (!accessToken && !code) {
      return;
    }
    try {
      if (accessToken && refreshToken) {
        const { data, error } = await supabase.auth.setSession({
          access_token: accessToken,
          refresh_token: refreshToken,
        });
        if (error) {
          throw error;
        }
        if (data.session) {
          await connectWithSession(data.session, {
            silent: false,
            pendingCaseId: draft.pendingCaseId ?? null,
          });
        }
      } else if (code) {
        const { data, error } = await supabase.auth.exchangeCodeForSession(url);
        if (error) {
          throw error;
        }
        if (data.session) {
          await connectWithSession(data.session, {
            silent: false,
            pendingCaseId: draft.pendingCaseId ?? null,
          });
        }
      }
      setLoginMessage("Signed in. Restoring your pilot workspace…");
    } catch (error) {
      setLoginMessage(errorMessage(error));
    }
  }

  async function sendMagicLink() {
    if (!supabase) {
      Alert.alert(
        "Supabase not configured",
        "Set EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY for the mobile pilot build.",
      );
      return;
    }
    if (!authEmail.trim()) {
      Alert.alert("Email required", "Enter the invited email address for this pilot.");
      return;
    }
    setSubmitting(true);
    setLoginMessage(null);
    try {
      const { error } = await supabase.auth.signInWithOtp({
        email: authEmail.trim().toLowerCase(),
        options: { emailRedirectTo: MOBILE_REDIRECT_URL },
      });
      if (error) {
        throw error;
      }
      setLoginMessage("Magic link sent. Open it on this device to finish sign-in.");
    } catch (error) {
      setLoginMessage(errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function connectWithSession(
    nextSession: Session,
    options: { silent: boolean; pendingCaseId: string | null },
  ) {
    const me = await fetchJson<Identity>(`${apiBaseUrl}/auth/me`, {
      headers: authHeaders(nextSession.access_token),
    });
    setIdentity(me);
    await Promise.all([
      refreshCases(nextSession.access_token),
      loadSites(nextSession.access_token),
    ]);
    if (options.pendingCaseId) {
      await openCase(options.pendingCaseId, {
        accessToken: nextSession.access_token,
        preservePending: true,
      });
      if (!options.silent) {
        setSyncMessage(`Restored pending case ${options.pendingCaseId}.`);
      }
    } else if (!options.silent) {
      setSyncMessage("Signed in.");
    }
  }

  async function refreshCases(accessToken = session?.access_token) {
    if (!accessToken) {
      return;
    }
    try {
      const payload = await fetchJson<{ items: CaseSummary[] }>(`${apiBaseUrl}/cases`, {
        headers: authHeaders(accessToken),
      });
      startTransition(() => {
        setRecentCases(payload.items);
      });
    } catch (error) {
      setSyncMessage(errorMessage(error));
    }
  }

  async function loadSites(accessToken = session?.access_token) {
    if (!accessToken) {
      return;
    }
    setLoadingCatalog(true);
    try {
      const payload = await fetchJson<{ items: SiteRecord[] }>(`${apiBaseUrl}/catalog/sites`, {
        headers: authHeaders(accessToken),
      });
      setSites(payload.items);
    } catch (error) {
      setSyncMessage(errorMessage(error));
    } finally {
      setLoadingCatalog(false);
    }
  }

  async function loadAssets(accessToken = session?.access_token) {
    if (!accessToken || !draft.siteId) {
      return;
    }
    try {
      const params = new URLSearchParams({ site_id: draft.siteId });
      if (assetQuery.trim()) {
        params.set("q", assetQuery.trim());
      }
      const payload = await fetchJson<{ items: AssetRecord[] }>(
        `${apiBaseUrl}/catalog/assets?${params.toString()}`,
        {
          headers: authHeaders(accessToken),
        },
      );
      setAssets(payload.items);
    } catch (error) {
      setSyncMessage(errorMessage(error));
    }
  }

  async function capture(mediaType: "image" | "video") {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Camera required", "Grant camera access to capture panel evidence.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes:
        mediaType === "image"
          ? ImagePicker.MediaTypeOptions.Images
          : ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: mediaType === "image",
      quality: 0.85,
      videoMaxDuration: 12,
    });
    if (result.canceled || result.assets.length === 0) {
      return;
    }
    const asset = result.assets[0];
    setCaptureHints([]);
    setDraft((current) => ({
      ...current,
      selectedMedia: {
        uri: asset.uri,
        name: asset.fileName ?? `capture.${mediaType === "image" ? "jpg" : "mp4"}`,
        mimeType: asset.mimeType ?? (mediaType === "image" ? "image/jpeg" : "video/mp4"),
        mediaType,
      },
    }));
  }

  async function analyzeCase() {
    if (!identity || !session) {
      Alert.alert("Sign in required", "Sign in before creating a triage case.");
      return;
    }
    if (!draft.selectedMedia) {
      Alert.alert("Capture required", "Capture a photo or short clip before analysis.");
      return;
    }
    if (!draft.siteId || !draft.assetId) {
      Alert.alert("Missing case details", "Select a site and asset before analysis.");
      return;
    }
    setSubmitting(true);
    setCaptureHints([]);
    setSyncMessage("Uploading evidence and requesting analysis…");
    let caseId = draft.pendingCaseId;
    try {
      if (!caseId) {
        const created = await fetchJson<CaseDetail>(`${apiBaseUrl}/cases`, {
          method: "POST",
          headers: authHeaders(session.access_token),
          body: JSON.stringify({
            site_id: draft.siteId,
            asset_id: draft.assetId,
            question: emptyToNull(draft.question),
            operator_context: emptyToNull(draft.operatorContext),
            expected_state_label: emptyToNull(draft.expectedStateLabel),
          }),
        });
        caseId = created.case_id;
        setDraft((current) => ({ ...current, pendingCaseId: created.case_id }));
      }

      const form = new FormData();
      form.append(
        "file",
        {
          uri: draft.selectedMedia.uri,
          name: draft.selectedMedia.name,
          type: draft.selectedMedia.mimeType,
        } as never,
      );
      const analyzed = await fetchJson<CaseDetail>(`${apiBaseUrl}/cases/${caseId}/analyze`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${session.access_token}`,
        },
        body: form,
      });
      setSelectedCase(analyzed);
      rememberRecentAsset(draft.assetId);
      setDraft((current) => ({
        ...DEFAULT_DRAFT,
        siteId: current.siteId,
        assetId: current.assetId,
        pendingCaseId: null,
      }));
      setSyncMessage("Analysis complete.");
      await refreshCases();
    } catch (error) {
      const details = structuredError(error);
      if (details.hints) {
        setCaptureHints(details.hints);
      }
      setDraft((current) => ({
        ...current,
        pendingCaseId: caseId ?? current.pendingCaseId ?? null,
      }));
      setSyncMessage(errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  }

  function rememberRecentAsset(assetId: string) {
    setRecentAssetIds((current) => [assetId, ...current.filter((item) => item !== assetId)].slice(0, 5));
  }

  async function submitFeedback(labels: string[], comment?: string) {
    if (!selectedCase || !session) {
      return;
    }
    setSubmitting(true);
    try {
      const updated = await fetchJson<CaseDetail>(
        `${apiBaseUrl}/cases/${selectedCase.case_id}/feedback`,
        {
          method: "POST",
          headers: authHeaders(session.access_token),
          body: JSON.stringify({ labels, comment }),
        },
      );
      setSelectedCase(updated);
      await refreshCases();
    } catch (error) {
      Alert.alert("Feedback failed", errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function openCase(
    caseId: string,
    options?: { accessToken?: string; preservePending?: boolean },
  ) {
    const accessToken = options?.accessToken ?? session?.access_token;
    if (!accessToken) {
      return;
    }
    try {
      const detail = await fetchJson<CaseDetail>(`${apiBaseUrl}/cases/${caseId}`, {
        headers: authHeaders(accessToken),
      });
      setSelectedCase(detail);
      if (!options?.preservePending && detail.analysis) {
        setDraft((current) => ({ ...current, pendingCaseId: null }));
      }
      if (!detail.analysis) {
        setDraft((current) => ({ ...current, pendingCaseId: caseId }));
        setSyncMessage(pendingCaseStatusMessage(detail.status));
      }
    } catch (error) {
      if (options?.preservePending) {
        setDraft((current) => ({ ...current, pendingCaseId: caseId }));
        setSyncMessage(
          "Pending case was restored locally. Retry the upload once the network is stable.",
        );
        return;
      }
      Alert.alert("Unable to load case", errorMessage(error));
    }
  }

  async function clearPendingDraft() {
    setDraft((current) => ({ ...current, pendingCaseId: null }));
    setSyncMessage("Pending analysis marker cleared.");
  }

  async function signOut() {
    if (supabase) {
      await supabase.auth.signOut();
    }
    setIdentity(null);
    setSession(null);
    setSelectedCase(null);
    setRecentCases([]);
    setSites([]);
    setAssets([]);
    setLoginMessage("Signed out.");
  }

  async function saveDebugBaseUrl() {
    const next = debugApiBaseUrl.trim() || DEFAULT_API_BASE_URL;
    setApiBaseUrl(next);
    if (DEBUG_OVERRIDE_ENABLED) {
      await AsyncStorage.setItem(DEV_API_BASE_KEY, next);
    }
    setSyncMessage(`Developer API override set to ${next}.`);
    setDebugVisible(false);
    if (session) {
      await connectWithSession(session, {
        silent: true,
        pendingCaseId: draft.pendingCaseId ?? null,
      });
    }
  }

  async function resetDebugBaseUrl() {
    setApiBaseUrl(DEFAULT_API_BASE_URL);
    setDebugApiBaseUrl(DEFAULT_API_BASE_URL);
    if (DEBUG_OVERRIDE_ENABLED) {
      await AsyncStorage.removeItem(DEV_API_BASE_KEY);
    }
    setSyncMessage("Developer API override cleared.");
    setDebugVisible(false);
  }

  function onBuildLabelPress() {
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

  if (loadingSession) {
    return (
      <CenteredShell>
        <ActivityIndicator color="#F08C2E" />
        <Text style={styles.loadingText}>Loading pilot workspace…</Text>
      </CenteredShell>
    );
  }

  if (!identity) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <StatusBar style="light" />
        <ScrollView contentContainerStyle={styles.authScroll}>
          <View style={styles.authHero}>
            <Text style={styles.eyebrow}>Field-Tech Pilot</Text>
            <Text style={styles.heroTitle}>Capture a panel, not a token</Text>
            <Text style={styles.heroBody}>
              Sign in with your invited email, capture a panel photo or short clip on-site,
              and get likely issue candidates plus grounded next steps.
            </Text>
          </View>
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Sign in with magic link</Text>
            <Field
              label="Invited email"
              value={authEmail}
              onChangeText={setAuthEmail}
              placeholder="tech@example.com"
              autoCapitalize="none"
              keyboardType="email-address"
            />
            <Text style={styles.metaText}>
              The pilot build is preconfigured for one backend environment. Magic links only.
            </Text>
            {loginMessage ? <Text style={styles.metaText}>{loginMessage}</Text> : null}
            <PrimaryButton title="Send magic link" loading={submitting} onPress={sendMagicLink} />
            {!supabase ? (
              <Text style={styles.captureWarningText}>
                Supabase mobile auth is not configured for this build.
              </Text>
            ) : null}
          </View>

          <Pressable onPress={onBuildLabelPress}>
            <Text style={styles.buildLabel}>pilot build · tap 5x for developer override</Text>
          </Pressable>

          {debugVisible ? (
            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Developer override</Text>
              <Field
                label="API base URL"
                value={debugApiBaseUrl}
                onChangeText={setDebugApiBaseUrl}
                placeholder={DEFAULT_API_BASE_URL}
                autoCapitalize="none"
              />
              <View style={styles.captureRow}>
                <SecondaryAction title="Save override" onPress={() => void saveDebugBaseUrl()} />
                <SecondaryAction title="Reset" onPress={() => void resetDebugBaseUrl()} />
              </View>
            </View>
          ) : null}
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.appScroll}>
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>Technician Workspace</Text>
            <Text style={styles.headerTitle}>{identity.display_name}</Text>
            <Text style={styles.metaText}>
              {identity.role} · {identity.organization_id ?? "service"}
            </Text>
          </View>
          <Pressable style={styles.secondaryButton} onPress={() => void signOut()}>
            <Text style={styles.secondaryButtonText}>Sign out</Text>
          </Pressable>
        </View>

        <View style={styles.banner}>
          <Text style={styles.bannerTitle}>Safety and trust</Text>
          <Text style={styles.bannerBody}>
            This tool surfaces likely issue candidates and next inspection steps. It does not
            replace lockout/tagout, measurement, or qualified judgment.
          </Text>
        </View>

        {pendingLabel ? (
          <View style={styles.pendingBanner}>
            <Text style={styles.pendingText}>{pendingLabel}</Text>
            <View style={styles.pendingActions}>
              <SecondaryAction
                title="Open pending case"
                onPress={() => {
                  if (draft.pendingCaseId) {
                    void openCase(draft.pendingCaseId);
                  }
                }}
              />
              <SecondaryAction title="Clear marker" onPress={() => void clearPendingDraft()} />
            </View>
          </View>
        ) : null}

        <View style={styles.card}>
          <View style={styles.inlineHeader}>
            <Text style={styles.sectionTitle}>Case setup</Text>
            <Pressable onPress={() => void loadSites()}>
              <Text style={styles.linkText}>Refresh catalog</Text>
            </Pressable>
          </View>
          <Text style={styles.metaText}>
            Capture the whole panel, keep indicator lights readable, and keep short clips under
            12 seconds.
          </Text>

          <Field
            label="Find site"
            value={siteQuery}
            onChangeText={setSiteQuery}
            placeholder="Search site name or code"
          />
          {loadingCatalog ? <Text style={styles.metaText}>Loading sites…</Text> : null}
          <SelectionList
            emptyMessage="No sites available yet in this organization."
            items={filteredSites}
            selectedId={draft.siteId}
            label={(site) => `${site.name}${site.code ? ` · ${site.code}` : ""}`}
            subtitle={(site) => site.site_id}
            onSelect={(site) => {
              setDraft((current) => ({
                ...current,
                siteId: site.site_id,
                assetId: current.siteId === site.site_id ? current.assetId : "",
              }));
              setAssetQuery("");
            }}
          />

          {selectedSite ? (
            <>
              <Field
                label="Find asset"
                value={assetQuery}
                onChangeText={setAssetQuery}
                placeholder="Search display name, panel family, or panel ID"
              />
              {recentAssets.length ? (
                <View style={styles.recentStrip}>
                  <Text style={styles.metaText}>Recent assets</Text>
                  <View style={styles.chipRow}>
                    {recentAssets.map((asset) => (
                      <Pressable
                        key={asset.asset_id}
                        style={styles.chip}
                        onPress={() =>
                          setDraft((current) => ({
                            ...current,
                            assetId: asset.asset_id,
                            siteId: asset.site_id,
                          }))
                        }
                      >
                        <Text style={styles.chipText}>{asset.display_name}</Text>
                      </Pressable>
                    ))}
                  </View>
                </View>
              ) : null}
              <SelectionList
                emptyMessage="No assets found for this site yet."
                items={assets}
                selectedId={draft.assetId}
                label={(asset) => asset.display_name}
                subtitle={(asset) =>
                  `${asset.asset_id} · ${asset.panel_family}${asset.panel_id ? ` · ${asset.panel_id}` : ""}`
                }
                onSelect={(asset) => {
                  setDraft((current) => ({ ...current, assetId: asset.asset_id }));
                  rememberRecentAsset(asset.asset_id);
                }}
              />
            </>
          ) : (
            <Text style={styles.metaText}>Select a site before choosing an asset.</Text>
          )}

          {selectedAsset ? (
            <View style={styles.previewCard}>
              <Text style={styles.previewTitle}>{selectedAsset.display_name}</Text>
              <Text style={styles.metaText}>
                {selectedAsset.asset_id} · {selectedAsset.panel_family} ·{" "}
                {selectedAsset.panel_id ?? "panel id not set"}
              </Text>
            </View>
          ) : null}

          <Field
            label="What are you seeing?"
            value={draft.question}
            onChangeText={(value) => setDraftValue(setDraft, "question", value)}
            multiline
          />
          <Field
            label="Operator context"
            value={draft.operatorContext}
            onChangeText={(value) => setDraftValue(setDraft, "operatorContext", value)}
            multiline
          />
          <Field
            label="Expected state label"
            value={draft.expectedStateLabel}
            onChangeText={(value) => setDraftValue(setDraft, "expectedStateLabel", value)}
            placeholder="optional"
          />

          <View style={styles.captureRow}>
            <SecondaryAction title="Capture photo" onPress={() => void capture("image")} />
            <SecondaryAction title="Capture short clip" onPress={() => void capture("video")} />
          </View>

          {draft.selectedMedia ? (
            <View style={styles.previewCard}>
              <Text style={styles.previewTitle}>
                Ready to upload · {draft.selectedMedia.mediaType.toUpperCase()}
              </Text>
              {draft.selectedMedia.mediaType === "image" ? (
                <Image source={{ uri: draft.selectedMedia.uri }} style={styles.previewImage} />
              ) : (
                <View style={styles.videoPlaceholder}>
                  <Text style={styles.videoPlaceholderText}>Short clip selected</Text>
                </View>
              )}
              <Text style={styles.metaText}>{draft.selectedMedia.name}</Text>
            </View>
          ) : null}

          {captureHints.length > 0 ? (
            <View style={styles.captureWarning}>
              <Text style={styles.captureWarningTitle}>Retake guidance</Text>
              {captureHints.map((hint) => (
                <Text key={hint} style={styles.captureWarningText}>
                  • {hint}
                </Text>
              ))}
            </View>
          ) : null}

          <PrimaryButton title="Analyze case" loading={submitting} onPress={() => void analyzeCase()} />
          {syncMessage ? <Text style={styles.metaText}>{syncMessage}</Text> : null}
        </View>

        <View style={styles.card}>
          <View style={styles.inlineHeader}>
            <Text style={styles.sectionTitle}>Recent cases</Text>
            <Pressable onPress={() => void refreshCases()}>
              <Text style={styles.linkText}>Refresh</Text>
            </Pressable>
          </View>
          {recentCases.length === 0 ? (
            <Text style={styles.metaText}>No cases yet in this pilot workspace.</Text>
          ) : (
            recentCases.map((item) => (
              <Pressable
                key={item.case_id}
                style={styles.caseRow}
                onPress={() => void openCase(item.case_id)}
              >
                <View style={{ flex: 1 }}>
                  <Text style={styles.caseTitle}>
                    {item.site_id} · {item.asset_id}
                  </Text>
                  <Text style={styles.metaText}>
                    {item.status} · {item.top_issue_class ?? "awaiting analysis"}
                  </Text>
                </View>
                <Text style={styles.caseBadge}>
                  {item.escalation_recommendation ?? "pending"}
                </Text>
              </Pressable>
            ))
          )}
        </View>

        {selectedCase?.analysis ? (
          <View style={styles.resultShell}>
            <View style={styles.resultHeader}>
              <Text style={styles.sectionTitle}>Case result</Text>
              <Text style={styles.resultStatus}>{selectedCase.status}</Text>
            </View>
            <View style={styles.banner}>
              <Text style={styles.bannerTitle}>
                {selectedCase.analysis.visual_evidence_status.replaceAll("_", " ")}
              </Text>
              <Text style={styles.bannerBody}>
                {selectedCase.analysis.uncertainty_summary ?? "Review the ranked evidence below."}
              </Text>
            </View>

            <CardSection title="Likely issue candidates">
              {selectedCase.analysis.issue_candidates.map((item) => (
                <View key={item.issue_class} style={styles.resultRow}>
                  <Text style={styles.resultPrimary}>
                    {item.issue_class} · {Math.round(item.confidence * 100)}%
                  </Text>
                  <Text style={styles.resultSecondary}>{item.rationale}</Text>
                </View>
              ))}
            </CardSection>

            <CardSection title="State assessment">
              <Text style={styles.resultPrimary}>
                {selectedCase.analysis.state_assessment.matched_state_label ?? "No state match"}
              </Text>
              <Text style={styles.resultSecondary}>
                {selectedCase.analysis.state_assessment.summary}
              </Text>
            </CardSection>

            <CardSection title="Next inspection steps">
              {selectedCase.analysis.next_steps.map((step) => (
                <View key={step.step} style={styles.resultRow}>
                  <Text style={styles.resultPrimary}>{step.step}</Text>
                  <Text style={styles.resultSecondary}>
                    {step.citations[0]?.title ?? "No citation"} ·{" "}
                    {Math.round(step.confidence * 100)}%
                  </Text>
                </View>
              ))}
            </CardSection>

            <CardSection title="Similar cases">
              {selectedCase.analysis.similar_incidents.map((incident) => (
                <View key={incident.title} style={styles.resultRow}>
                  <Text style={styles.resultPrimary}>{incident.title}</Text>
                  <Text style={styles.resultSecondary}>
                    {incident.fix_summary} · {Math.round(incident.similarity * 100)}%
                  </Text>
                </View>
              ))}
            </CardSection>

            <CardSection title="Escalation">
              <Text style={styles.resultPrimary}>
                {selectedCase.analysis.escalation_recommendation.replaceAll("_", " ")}
              </Text>
            </CardSection>

            <CardSection title="Safety notices">
              {selectedCase.analysis.safety_notices.map((notice) => (
                <Text key={notice} style={styles.resultSecondary}>
                  • {notice}
                </Text>
              ))}
            </CardSection>

            <View style={styles.captureRow}>
              <SecondaryAction
                title="Helpful"
                onPress={() =>
                  void submitFeedback(["helpful"], "This result shortened the inspection loop.")
                }
              />
              <SecondaryAction
                title="Escalated anyway"
                onPress={() =>
                  void submitFeedback(
                    ["escalated_anyway"],
                    "Escalated because visible evidence was not enough.",
                  )
                }
              />
            </View>
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function CardSection(props: { title: string; children: ReactNode }) {
  return (
    <View style={styles.card}>
      <Text style={styles.sectionTitle}>{props.title}</Text>
      {props.children}
    </View>
  );
}

function SelectionList<T extends { site_id?: string; asset_id?: string }>(props: {
  items: T[];
  selectedId: string;
  emptyMessage: string;
  label: (item: T) => string;
  subtitle: (item: T) => string;
  onSelect: (item: T) => void;
}) {
  if (props.items.length === 0) {
    return <Text style={styles.metaText}>{props.emptyMessage}</Text>;
  }
  return (
    <View style={styles.selectionList}>
      {props.items.map((item) => {
        const id = ("site_id" in item ? item.site_id : item.asset_id) ?? "";
        const selected = props.selectedId === id;
        return (
          <Pressable
            key={id}
            style={[styles.selectionRow, selected ? styles.selectionRowActive : null]}
            onPress={() => props.onSelect(item)}
          >
            <Text style={styles.selectionTitle}>{props.label(item)}</Text>
            <Text style={styles.metaText}>{props.subtitle(item)}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function Field(props: {
  label: string;
  value: string;
  onChangeText: (value: string) => void;
  placeholder?: string;
  multiline?: boolean;
  autoCapitalize?: "none" | "sentences" | "words" | "characters";
  secureTextEntry?: boolean;
  keyboardType?:
    | "default"
    | "email-address"
    | "numeric"
    | "phone-pad"
    | "url";
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{props.label}</Text>
      <TextInput
        value={props.value}
        onChangeText={props.onChangeText}
        placeholder={props.placeholder}
        placeholderTextColor="#6E7C81"
        style={[styles.input, props.multiline ? styles.textArea : null]}
        multiline={props.multiline}
        autoCapitalize={props.autoCapitalize}
        secureTextEntry={props.secureTextEntry}
        keyboardType={props.keyboardType}
      />
    </View>
  );
}

function PrimaryButton(props: {
  title: string;
  onPress: () => void;
  loading?: boolean;
}) {
  return (
    <Pressable
      style={[styles.primaryButton, props.loading ? styles.disabledButton : null]}
      onPress={props.onPress}
      disabled={props.loading}
    >
      {props.loading ? (
        <ActivityIndicator color="#fff9f0" />
      ) : (
        <Text style={styles.primaryButtonText}>{props.title}</Text>
      )}
    </Pressable>
  );
}

function SecondaryAction(props: { title: string; onPress: () => void }) {
  return (
    <Pressable style={styles.secondaryAction} onPress={props.onPress}>
      <Text style={styles.secondaryActionText}>{props.title}</Text>
    </Pressable>
  );
}

function CenteredShell(props: { children: ReactNode }) {
  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <View style={styles.centeredShell}>{props.children}</View>
    </SafeAreaView>
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

function emptyToNull(value: string): string | null {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function structuredError(error: unknown): { message: string; hints?: string[] } {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "object" && detail !== null) {
      const message =
        "message" in detail && typeof detail.message === "string"
          ? detail.message
          : "The request could not be completed.";
      const hints =
        "hints" in detail && Array.isArray(detail.hints)
          ? detail.hints.filter((item): item is string => typeof item === "string")
          : undefined;
      return { message, hints };
    }
    if (typeof detail === "string") {
      return { message: detail };
    }
  }
  return { message: "The request could not be completed." };
}

function errorMessage(error: unknown): string {
  return structuredError(error).message;
}

function extractUrlParams(url: string): URLSearchParams {
  const [beforeHash, hash = ""] = url.split("#", 2);
  const query = beforeHash.includes("?") ? beforeHash.split("?")[1] ?? "" : "";
  const params = new URLSearchParams(query);
  const hashParams = new URLSearchParams(hash);
  hashParams.forEach((value, key) => {
    params.set(key, value);
  });
  return params;
}

function setDraftValue(
  setDraft: Dispatch<SetStateAction<DraftCase>>,
  key: "siteId" | "assetId" | "question" | "operatorContext" | "expectedStateLabel",
  value: string,
) {
  setDraft((current) => ({ ...current, [key]: value }));
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#0E181D",
  },
  centeredShell: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    paddingHorizontal: 24,
  },
  loadingText: {
    color: "#E9E1D0",
    fontSize: 16,
  },
  authScroll: {
    padding: 20,
    gap: 18,
  },
  appScroll: {
    padding: 18,
    gap: 16,
  },
  authHero: {
    paddingVertical: 12,
    gap: 10,
  },
  eyebrow: {
    color: "#F08C2E",
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 2,
    textTransform: "uppercase",
  },
  heroTitle: {
    color: "#FFF5E5",
    fontSize: 34,
    fontWeight: "700",
    lineHeight: 36,
  },
  heroBody: {
    color: "#C8D1D5",
    fontSize: 16,
    lineHeight: 24,
  },
  header: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 12,
  },
  headerTitle: {
    color: "#FFF5E5",
    fontSize: 28,
    fontWeight: "700",
  },
  card: {
    borderRadius: 22,
    backgroundColor: "#142229",
    padding: 16,
    gap: 12,
    borderWidth: 1,
    borderColor: "#23363D",
  },
  banner: {
    borderRadius: 18,
    backgroundColor: "#1B2C34",
    padding: 14,
    gap: 6,
    borderWidth: 1,
    borderColor: "#2E4650",
  },
  bannerTitle: {
    color: "#F4B15C",
    fontSize: 14,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1.2,
  },
  bannerBody: {
    color: "#D6DEE1",
    fontSize: 14,
    lineHeight: 20,
  },
  pendingBanner: {
    borderRadius: 18,
    backgroundColor: "#332416",
    borderWidth: 1,
    borderColor: "#6B4725",
    padding: 14,
    gap: 10,
  },
  pendingText: {
    color: "#FFD7A8",
    fontSize: 14,
    lineHeight: 20,
  },
  pendingActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  field: {
    gap: 6,
  },
  fieldLabel: {
    color: "#E5E9EA",
    fontSize: 13,
    fontWeight: "700",
  },
  input: {
    borderRadius: 16,
    backgroundColor: "#0D1519",
    borderWidth: 1,
    borderColor: "#2C3D43",
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: "#FFF5E5",
    fontSize: 15,
  },
  textArea: {
    minHeight: 92,
    textAlignVertical: "top",
  },
  metaText: {
    color: "#AEBBC0",
    fontSize: 13,
    lineHeight: 18,
  },
  sectionTitle: {
    color: "#FFF5E5",
    fontSize: 18,
    fontWeight: "700",
  },
  inlineHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  primaryButton: {
    borderRadius: 16,
    backgroundColor: "#D6761E",
    paddingVertical: 14,
    alignItems: "center",
    justifyContent: "center",
  },
  primaryButtonText: {
    color: "#FFF8F0",
    fontSize: 15,
    fontWeight: "700",
  },
  secondaryButton: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#3B4C53",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  secondaryButtonText: {
    color: "#D6DEE1",
    fontWeight: "700",
  },
  secondaryAction: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#38515A",
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: "#112027",
  },
  secondaryActionText: {
    color: "#DCE5E8",
    fontWeight: "700",
  },
  disabledButton: {
    opacity: 0.7,
  },
  captureRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  previewCard: {
    borderRadius: 16,
    backgroundColor: "#0D1519",
    borderWidth: 1,
    borderColor: "#2B3E46",
    padding: 12,
    gap: 8,
  },
  previewTitle: {
    color: "#F8F0E0",
    fontSize: 15,
    fontWeight: "700",
  },
  previewImage: {
    width: "100%",
    height: 180,
    borderRadius: 14,
    backgroundColor: "#243238",
  },
  videoPlaceholder: {
    height: 120,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#1A2A30",
  },
  videoPlaceholderText: {
    color: "#D7E1E4",
    fontWeight: "700",
  },
  captureWarning: {
    borderRadius: 16,
    padding: 12,
    borderWidth: 1,
    borderColor: "#7B5230",
    backgroundColor: "#342516",
    gap: 6,
  },
  captureWarningTitle: {
    color: "#FFD7A8",
    fontWeight: "700",
  },
  captureWarningText: {
    color: "#F6D1AC",
    lineHeight: 18,
  },
  selectionList: {
    gap: 8,
  },
  selectionRow: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#27383F",
    backgroundColor: "#101A1F",
    padding: 12,
    gap: 4,
  },
  selectionRowActive: {
    borderColor: "#D6761E",
    backgroundColor: "#1F241F",
  },
  selectionTitle: {
    color: "#FFF5E5",
    fontWeight: "700",
  },
  recentStrip: {
    gap: 8,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  chip: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: "#24363C",
  },
  chipText: {
    color: "#E4ECEE",
    fontWeight: "600",
  },
  linkText: {
    color: "#F08C2E",
    fontWeight: "700",
  },
  caseRow: {
    flexDirection: "row",
    gap: 12,
    alignItems: "center",
    borderRadius: 16,
    backgroundColor: "#0F181D",
    borderWidth: 1,
    borderColor: "#23363D",
    padding: 12,
    marginBottom: 10,
  },
  caseTitle: {
    color: "#FFF5E5",
    fontWeight: "700",
  },
  caseBadge: {
    color: "#F3BF7E",
    fontSize: 12,
    textTransform: "uppercase",
  },
  resultShell: {
    gap: 12,
  },
  resultHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  resultStatus: {
    color: "#F3BF7E",
    fontWeight: "700",
    textTransform: "uppercase",
  },
  resultRow: {
    gap: 4,
    marginBottom: 8,
  },
  resultPrimary: {
    color: "#F8F0E0",
    fontSize: 15,
    fontWeight: "700",
    lineHeight: 20,
  },
  resultSecondary: {
    color: "#C4D0D4",
    lineHeight: 19,
  },
  buildLabel: {
    color: "#73858C",
    fontSize: 12,
    textAlign: "center",
  },
});
