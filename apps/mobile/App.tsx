import type { ReactNode } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { startTransition, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

type Identity = {
  principal: string;
  principal_type: string;
  role: string;
  user_id?: string | null;
  organization_id?: string | null;
  display_name: string;
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
  panelFamily: string;
  panelId: string;
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

const SESSION_KEY = "mtc.mobile.session";
const DRAFT_KEY = "mtc.mobile.draft";
const DEFAULT_DRAFT: DraftCase = {
  siteId: "",
  assetId: "",
  panelFamily: "electrical_panel_family_a",
  panelId: "",
  question: "",
  operatorContext: "",
  expectedStateLabel: "",
  pendingCaseId: null,
  selectedMedia: null,
};

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [token, setToken] = useState("");
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [draft, setDraft] = useState<DraftCase>(DEFAULT_DRAFT);
  const [recentCases, setRecentCases] = useState<CaseSummary[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseDetail | null>(null);
  const [loadingSession, setLoadingSession] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);
  const [captureHints, setCaptureHints] = useState<string[]>([]);

  useEffect(() => {
    void loadPersistedState();
  }, []);

  useEffect(() => {
    if (identity !== null) {
      void AsyncStorage.setItem(
        SESSION_KEY,
        JSON.stringify({ apiBaseUrl, token, identity }),
      );
    }
  }, [apiBaseUrl, identity, token]);

  useEffect(() => {
    if (!loadingSession) {
      void AsyncStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
    }
  }, [draft, loadingSession]);

  const pendingLabel = useMemo(() => {
    if (!draft.pendingCaseId) {
      return null;
    }
    return `Analysis pending for ${draft.pendingCaseId}. Retry upload when connectivity stabilizes.`;
  }, [draft.pendingCaseId]);

  async function loadPersistedState() {
    try {
      const [sessionRaw, draftRaw] = await Promise.all([
        AsyncStorage.getItem(SESSION_KEY),
        AsyncStorage.getItem(DRAFT_KEY),
      ]);
      let restoredDraft = DEFAULT_DRAFT;
      if (draftRaw) {
        restoredDraft = { ...DEFAULT_DRAFT, ...(JSON.parse(draftRaw) as DraftCase) };
        setDraft(restoredDraft);
      }
      if (sessionRaw) {
        const session = JSON.parse(sessionRaw) as {
          apiBaseUrl: string;
          token: string;
          identity: Identity;
        };
        setApiBaseUrl(session.apiBaseUrl);
        setToken(session.token);
        await connectWithCredentials(session.apiBaseUrl, session.token, {
          silent: true,
          pendingCaseId: restoredDraft.pendingCaseId ?? null,
        });
      }
    } catch (error) {
      setSyncMessage(errorMessage(error));
      await AsyncStorage.removeItem(SESSION_KEY);
      setIdentity(null);
      setToken("");
    } finally {
      setLoadingSession(false);
    }
  }

  async function connect() {
    setSubmitting(true);
    setSyncMessage(null);
    try {
      await connectWithCredentials(apiBaseUrl, token, {
        silent: false,
        pendingCaseId: draft.pendingCaseId ?? null,
      });
    } catch (error) {
      Alert.alert("Connection failed", errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function connectWithCredentials(
    baseUrl: string,
    bearer: string,
    options: { silent: boolean; pendingCaseId: string | null },
  ) {
    const me = await fetchJson<Identity>(`${baseUrl}/auth/me`, {
      headers: authHeaders(bearer),
    });
    setIdentity(me);
    await refreshCases(baseUrl, bearer);
    if (options.pendingCaseId) {
      await openCase(options.pendingCaseId, {
        baseUrl,
        bearer,
        preservePending: true,
      });
      if (!options.silent) {
        setSyncMessage(`Restored pending case ${options.pendingCaseId}.`);
      }
    }
  }

  async function refreshCases(baseUrl = apiBaseUrl, bearer = token) {
    try {
      const payload = await fetchJson<{ items: CaseSummary[] }>(`${baseUrl}/cases`, {
        headers: authHeaders(bearer),
      });
      startTransition(() => {
        setRecentCases(payload.items);
      });
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
        mimeType:
          asset.mimeType ?? (mediaType === "image" ? "image/jpeg" : "video/mp4"),
        mediaType,
      },
    }));
  }

  async function analyzeCase() {
    if (!identity) {
      Alert.alert("Not connected", "Sign in before creating a triage case.");
      return;
    }
    if (!draft.selectedMedia) {
      Alert.alert("Capture required", "Capture a photo or short clip before analysis.");
      return;
    }
    if (!draft.siteId || !draft.assetId || !draft.panelFamily) {
      Alert.alert("Missing case details", "Site, asset, and panel family are required.");
      return;
    }
    setSubmitting(true);
    setCaptureHints([]);
    setSyncMessage("Uploading evidence and requesting analysis…");
    let caseId = draft.pendingCaseId;
    try {
      if (!caseId) {
        const created = await fetchJson<{ case_id: string }>(`${apiBaseUrl}/cases`, {
          method: "POST",
          headers: authHeaders(token),
          body: JSON.stringify({
            site_id: draft.siteId,
            asset_id: draft.assetId,
            panel_family: draft.panelFamily,
            panel_id: emptyToNull(draft.panelId),
            question: emptyToNull(draft.question),
            operator_context: emptyToNull(draft.operatorContext),
            expected_state_label: emptyToNull(draft.expectedStateLabel),
          }),
        });
        caseId = created.case_id;
        setDraft((current) => ({ ...current, pendingCaseId: created.case_id }));
      }

      const form = new FormData();
      form.append("file", {
        uri: draft.selectedMedia.uri,
        name: draft.selectedMedia.name,
        type: draft.selectedMedia.mimeType,
      } as never);
      const analyzed = await fetchJson<CaseDetail>(`${apiBaseUrl}/cases/${caseId}/analyze`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: form,
      });
      setSelectedCase(analyzed);
      setDraft((current) => ({
        ...DEFAULT_DRAFT,
        siteId: current.siteId,
        assetId: current.assetId,
        panelFamily: current.panelFamily,
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

  async function submitFeedback(labels: string[], comment?: string) {
    if (!selectedCase) {
      return;
    }
    setSubmitting(true);
    try {
      const updated = await fetchJson<CaseDetail>(
        `${apiBaseUrl}/cases/${selectedCase.case_id}/feedback`,
        {
          method: "POST",
          headers: authHeaders(token),
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
    options?: { baseUrl?: string; bearer?: string; preservePending?: boolean },
  ) {
    try {
      const detail = await fetchJson<CaseDetail>(
        `${options?.baseUrl ?? apiBaseUrl}/cases/${caseId}`,
        {
          headers: authHeaders(options?.bearer ?? token),
        },
      );
      setSelectedCase(detail);
      if (!options?.preservePending && detail.analysis) {
        setDraft((current) => ({ ...current, pendingCaseId: null }));
      }
      if (!detail.analysis) {
        setDraft((current) => ({ ...current, pendingCaseId: caseId }));
        setSyncMessage(
          detail.status === "pending_analysis"
            ? "This case is still waiting for a successful upload or analysis retry."
            : "This draft case still needs a successful media upload.",
        );
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
            <Text style={styles.heroTitle}>Mobile electrical-panel troubleshooting</Text>
            <Text style={styles.heroBody}>
              Capture a photo or short clip on-site, then get likely issue candidates, next
              inspection steps, similar cases, and whether to escalate.
            </Text>
          </View>
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Connect to pilot backend</Text>
            <Field
              label="API base URL"
              value={apiBaseUrl}
              onChangeText={setApiBaseUrl}
              placeholder="http://localhost:8000"
              autoCapitalize="none"
            />
            <Field
              label="Bearer token"
              value={token}
              onChangeText={setToken}
              placeholder="Paste invited-user token"
              autoCapitalize="none"
              secureTextEntry
            />
            <Text style={styles.metaText}>
              No offline inference. Capture is stored locally until upload succeeds.
            </Text>
            <PrimaryButton title="Connect" loading={submitting} onPress={connect} />
          </View>
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
          <Pressable
            style={styles.secondaryButton}
            onPress={async () => {
              await AsyncStorage.removeItem(SESSION_KEY);
              setIdentity(null);
              setToken("");
            }}
          >
            <Text style={styles.secondaryButtonText}>Switch token</Text>
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
          <Text style={styles.sectionTitle}>Case setup</Text>
          <Text style={styles.metaText}>
            Frame the whole panel, avoid glare, and keep indicator lights readable.
          </Text>
          <Field label="Site" value={draft.siteId} onChangeText={(value) => setDraftValue("siteId", value)} />
          <Field label="Asset" value={draft.assetId} onChangeText={(value) => setDraftValue("assetId", value)} />
          <Field
            label="Panel family"
            value={draft.panelFamily}
            onChangeText={(value) => setDraftValue("panelFamily", value)}
          />
          <Field label="Panel ID" value={draft.panelId} onChangeText={(value) => setDraftValue("panelId", value)} />
          <Field
            label="What are you seeing?"
            value={draft.question}
            onChangeText={(value) => setDraftValue("question", value)}
            multiline
          />
          <Field
            label="Operator context"
            value={draft.operatorContext}
            onChangeText={(value) => setDraftValue("operatorContext", value)}
            multiline
          />
          <Field
            label="Expected state label"
            value={draft.expectedStateLabel}
            onChangeText={(value) => setDraftValue("expectedStateLabel", value)}
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

            <CardSection title="Panel-state assessment">
              <Text style={styles.resultPrimary}>
                {selectedCase.analysis.state_assessment.matched_state_label ?? "No close reference state"}
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
                    {Math.round(step.confidence * 100)}% · {step.citations[0]?.title ?? "No citation"}
                  </Text>
                </View>
              ))}
            </CardSection>

            <CardSection title="Similar prior cases">
              {selectedCase.analysis.similar_incidents.map((incident) => (
                <View key={incident.title} style={styles.resultRow}>
                  <Text style={styles.resultPrimary}>{incident.title}</Text>
                  <Text style={styles.resultSecondary}>
                    {incident.fix_summary} · {Math.round(incident.similarity * 100)}%
                  </Text>
                </View>
              ))}
            </CardSection>

            <CardSection title="Escalation guidance">
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

            <View style={styles.feedbackRow}>
              <SecondaryAction title="Helpful" onPress={() => void submitFeedback(["helpful"])} />
              <SecondaryAction title="Not helpful" onPress={() => void submitFeedback(["not_helpful"])} />
              <SecondaryAction title="Escalated anyway" onPress={() => void submitFeedback(["escalated_anyway"], "Escalated despite the guided result.")} />
            </View>
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );

  function setDraftValue<Key extends keyof DraftCase>(key: Key, value: DraftCase[Key]) {
    setDraft((current) => ({ ...current, [key]: value }));
  }
}

function Field(props: {
  label: string;
  value: string;
  onChangeText: (value: string) => void;
  placeholder?: string;
  multiline?: boolean;
  autoCapitalize?: "none" | "sentences" | "words" | "characters";
  secureTextEntry?: boolean;
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.label}>{props.label}</Text>
      <TextInput
        value={props.value}
        onChangeText={props.onChangeText}
        placeholder={props.placeholder}
        placeholderTextColor="#7C8E96"
        multiline={props.multiline}
        autoCapitalize={props.autoCapitalize ?? "sentences"}
        secureTextEntry={props.secureTextEntry}
        style={[styles.input, props.multiline ? styles.multilineInput : null]}
      />
    </View>
  );
}

function PrimaryButton(props: { title: string; loading?: boolean; onPress: () => void }) {
  return (
    <Pressable
      onPress={props.onPress}
      disabled={props.loading}
      style={[styles.primaryButton, props.loading ? styles.buttonDisabled : null]}
    >
      {props.loading ? <ActivityIndicator color="#081116" /> : <Text style={styles.primaryButtonText}>{props.title}</Text>}
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

function CardSection(props: { title: string; children: ReactNode }) {
  return (
    <View style={styles.resultCard}>
      <Text style={styles.cardTitle}>{props.title}</Text>
      {props.children}
    </View>
  );
}

function CenteredShell(props: { children: ReactNode }) {
  return <SafeAreaView style={styles.centeredShell}>{props.children}</SafeAreaView>;
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
  return "The request did not complete. Retry when the connection is stable.";
}

function structuredError(error: unknown): { hints?: string[] } {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "object" && detail !== null && "hints" in detail) {
      return {
        hints: Array.isArray((detail as { hints: unknown }).hints)
          ? ((detail as { hints: string[] }).hints)
          : [],
      };
    }
  }
  return {};
}

function emptyToNull(value: string): string | null {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#0B151B",
  },
  authScroll: {
    padding: 24,
    gap: 18,
    backgroundColor: "#0B151B",
  },
  appScroll: {
    padding: 18,
    gap: 16,
    backgroundColor: "#0B151B",
  },
  centeredShell: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0B151B",
    gap: 12,
  },
  authHero: {
    padding: 24,
    borderRadius: 24,
    backgroundColor: "#13232C",
    gap: 12,
  },
  eyebrow: {
    color: "#F08C2E",
    textTransform: "uppercase",
    letterSpacing: 1.5,
    fontSize: 12,
    fontWeight: "700",
  },
  heroTitle: {
    color: "#F5F2E9",
    fontSize: 34,
    lineHeight: 38,
    fontWeight: "800",
  },
  heroBody: {
    color: "#C1CDD2",
    fontSize: 15,
    lineHeight: 22,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  headerTitle: {
    color: "#F5F2E9",
    fontSize: 28,
    fontWeight: "800",
  },
  card: {
    backgroundColor: "#12222A",
    borderRadius: 22,
    padding: 18,
    gap: 12,
    borderWidth: 1,
    borderColor: "#1D343F",
  },
  banner: {
    backgroundColor: "#1D2B12",
    borderRadius: 18,
    padding: 16,
    gap: 8,
    borderWidth: 1,
    borderColor: "#4F7433",
  },
  bannerTitle: {
    color: "#D9F59D",
    fontSize: 16,
    fontWeight: "700",
  },
  bannerBody: {
    color: "#E9F4D5",
    lineHeight: 20,
  },
  pendingBanner: {
    backgroundColor: "#3A2715",
    padding: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D6842D",
  },
  pendingText: {
    color: "#F8D4A2",
    lineHeight: 20,
  },
  pendingActions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 10,
  },
  sectionTitle: {
    color: "#F5F2E9",
    fontSize: 21,
    fontWeight: "800",
  },
  metaText: {
    color: "#A8B6BC",
    lineHeight: 19,
  },
  field: {
    gap: 6,
  },
  label: {
    color: "#D9E2E6",
    fontSize: 13,
    fontWeight: "700",
  },
  input: {
    backgroundColor: "#081116",
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: "#F5F2E9",
    borderWidth: 1,
    borderColor: "#1E353F",
  },
  multilineInput: {
    minHeight: 88,
    textAlignVertical: "top",
  },
  primaryButton: {
    backgroundColor: "#F08C2E",
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#081116",
    fontSize: 16,
    fontWeight: "800",
  },
  secondaryButton: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 14,
    backgroundColor: "#12222A",
    borderWidth: 1,
    borderColor: "#1E353F",
  },
  secondaryButtonText: {
    color: "#D9E2E6",
    fontWeight: "700",
  },
  secondaryAction: {
    flex: 1,
    minWidth: 0,
    backgroundColor: "#081116",
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#314853",
    alignItems: "center",
  },
  secondaryActionText: {
    color: "#E7EEF1",
    fontWeight: "700",
    textAlign: "center",
  },
  captureRow: {
    flexDirection: "row",
    gap: 10,
  },
  previewCard: {
    backgroundColor: "#081116",
    borderRadius: 16,
    padding: 12,
    gap: 10,
  },
  previewTitle: {
    color: "#F5F2E9",
    fontWeight: "700",
  },
  previewImage: {
    width: "100%",
    aspectRatio: 1.2,
    borderRadius: 14,
  },
  videoPlaceholder: {
    borderRadius: 14,
    minHeight: 160,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#15242D",
  },
  videoPlaceholderText: {
    color: "#C1CDD2",
    fontWeight: "700",
  },
  captureWarning: {
    backgroundColor: "#3E1C16",
    borderRadius: 16,
    padding: 14,
    gap: 4,
    borderWidth: 1,
    borderColor: "#A84D3A",
  },
  captureWarningTitle: {
    color: "#F9C4B7",
    fontWeight: "700",
  },
  captureWarningText: {
    color: "#FCE0D9",
    lineHeight: 19,
  },
  inlineHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  linkText: {
    color: "#F08C2E",
    fontWeight: "700",
  },
  caseRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#1A2E38",
  },
  caseTitle: {
    color: "#F5F2E9",
    fontWeight: "700",
  },
  caseBadge: {
    color: "#F08C2E",
    fontSize: 12,
    fontWeight: "800",
    textTransform: "uppercase",
    maxWidth: 120,
    textAlign: "right",
  },
  resultShell: {
    gap: 12,
  },
  resultHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  resultStatus: {
    color: "#F08C2E",
    fontWeight: "800",
    textTransform: "uppercase",
  },
  resultCard: {
    backgroundColor: "#12222A",
    borderRadius: 18,
    padding: 16,
    gap: 10,
  },
  cardTitle: {
    color: "#F5F2E9",
    fontWeight: "800",
    fontSize: 16,
  },
  resultRow: {
    gap: 4,
  },
  resultPrimary: {
    color: "#F5F2E9",
    fontWeight: "700",
    lineHeight: 21,
  },
  resultSecondary: {
    color: "#B7C5CB",
    lineHeight: 20,
  },
  feedbackRow: {
    flexDirection: "row",
    gap: 10,
  },
  loadingText: {
    color: "#C1CDD2",
  },
  buttonDisabled: {
    opacity: 0.7,
  },
});
