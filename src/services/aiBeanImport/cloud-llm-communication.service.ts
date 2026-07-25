import { AI_PROVIDER_ENUM } from '../../enums/settings/aiProvider';

export interface CloudLLMConfig {
  provider: AI_PROVIDER_ENUM;
  apiKey: string;
  model: string;
  baseUrl?: string; // for CUSTOM provider
}

export interface CloudLLMMessage {
  role: 'system' | 'user';
  content: string;
}

export interface CloudLLMResponse {
  content: string;
  model: string;
  usage?: { prompt_tokens: number; completion_tokens: number };
}

// ── Provider protocol config ─────────────────────────────────────────
//
// Each provider config defines URL, headers, request body shape, and
// response parsing. The shared sendCloudLLMPrompt function handles
// fetch, timeout, and error handling — protocol details stay here.

/**
 * Extra request-body fields tuning the model for this task (e.g. a low
 * temperature, a low reasoning effort). Provider-specific and fully formed —
 * see the model-options registry below.
 */
type ModelOptions = Record<string, unknown>;

/** Protocol-level config that describes how to talk to a specific LLM API. */
interface ProviderProtocol {
  readonly url: string;
  readonly headers: Record<string, string>;
  buildRequestBody(
    model: string,
    messages: CloudLLMMessage[],
    options: ModelOptions,
  ): object;
  parseResponse(body: unknown): CloudLLMResponse;
}

/** Read a nested property from an unknown value, returning undefined on miss. */
function dig(obj: unknown, ...keys: (string | number)[]): unknown {
  let cur = obj;
  for (const k of keys) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string | number, unknown>)[k];
  }
  return cur;
}

// ── OpenAI-compatible protocol ───────────────────────────────────────
//
// Used by OpenAI, Google (Gemini), Mistral, OpenRouter, and Custom.
// All share the same body format, endpoint, and response shape.
// Differences are limited to base URL and optional extra headers.

class OpenAICompatibleProtocol implements ProviderProtocol {
  readonly url: string;
  readonly headers: Record<string, string>;

  constructor(
    baseUrl: string,
    apiKey: string,
    extraHeaders?: Record<string, string>,
  ) {
    this.url = `${baseUrl}/chat/completions`;
    this.headers = {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      ...extraHeaders,
    };
  }

  buildRequestBody(
    model: string,
    messages: CloudLLMMessage[],
    options: ModelOptions,
  ): object {
    return {
      model,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      ...options,
    };
  }

  // API response shape is not validated — access defensively.
  parseResponse(body: unknown): CloudLLMResponse {
    const usage = dig(body, 'usage') as
      | { prompt_tokens: number; completion_tokens: number }
      | undefined;
    return {
      content: String(dig(body, 'choices', 0, 'message', 'content') ?? ''),
      model: String(dig(body, 'model') ?? ''),
      usage,
    };
  }
}

// ── Anthropic protocol ───────────────────────────────────────────────
//
// Different auth header, API version header, body format (system message
// extracted, max_tokens required), endpoint (/messages), and response
// shape (content[].text instead of choices[].message.content).

class AnthropicProtocol implements ProviderProtocol {
  readonly url: string;
  readonly headers: Record<string, string>;

  constructor(apiKey: string) {
    this.url = 'https://api.anthropic.com/v1/messages';
    this.headers = {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json',
    };
  }

  buildRequestBody(
    model: string,
    messages: CloudLLMMessage[],
    options: ModelOptions,
  ): object {
    const systemMsg = messages.find((m) => m.role === 'system');
    const userMsgs = messages.filter((m) => m.role !== 'system');
    return {
      model,
      max_tokens: 4096,
      ...options,
      system: systemMsg?.content ?? '',
      messages: userMsgs.map((m) => ({ role: m.role, content: m.content })),
    };
  }

  // API response shape is not validated — access defensively.
  parseResponse(body: unknown): CloudLLMResponse {
    const usage = dig(body, 'usage') as Record<string, unknown> | undefined;
    return {
      content: String(dig(body, 'content', 0, 'text') ?? ''),
      model: String(dig(body, 'model') ?? ''),
      usage: usage
        ? {
            prompt_tokens: Number(usage.input_tokens ?? 0),
            completion_tokens: Number(usage.output_tokens ?? 0),
          }
        : undefined,
    };
  }
}

// ── Factory ──────────────────────────────────────────────────────────

function createProtocol(config: CloudLLMConfig): ProviderProtocol {
  switch (config.provider) {
    case AI_PROVIDER_ENUM.ANTHROPIC:
      return new AnthropicProtocol(config.apiKey);

    case AI_PROVIDER_ENUM.OPENROUTER:
      return new OpenAICompatibleProtocol(
        'https://openrouter.ai/api/v1',
        config.apiKey,
        {
          'HTTP-Referer': 'https://beanconqueror.com',
          'X-OpenRouter-Title': 'Beanconqueror',
        },
      );

    case AI_PROVIDER_ENUM.CUSTOM:
      return new OpenAICompatibleProtocol(
        (config.baseUrl ?? '').replace(/\/+$/, ''),
        config.apiKey,
      );

    case AI_PROVIDER_ENUM.GOOGLE:
      return new OpenAICompatibleProtocol(
        'https://generativelanguage.googleapis.com/v1beta/openai',
        config.apiKey,
      );

    case AI_PROVIDER_ENUM.MISTRAL:
      return new OpenAICompatibleProtocol(
        'https://api.mistral.ai/v1',
        config.apiKey,
      );

    case AI_PROVIDER_ENUM.OPENAI:
    default:
      return new OpenAICompatibleProtocol(
        'https://api.openai.com/v1',
        config.apiKey,
      );
  }
}

// ── Model-options registry ───────────────────────────────────────────
//
// Bean import is a simple, factual extraction task, so we want the model as
// deterministic ("static") as possible. The right lever differs by model:
// older models honour a low `temperature`; the newest models (Claude Sonnet
// 5 / Opus 4.7+, OpenAI GPT-5.x, ...) have removed `temperature` and instead
// take a reasoning `effort`. There is no single cross-provider knob, so we
// store a fully-formed, provider-specific option set per known model and a
// per-provider default for models we do not yet know about.
//
// This registry is a QUALITY layer, never a correctness dependency: if an
// option set is wrong or stale, or a model is unknown, the request still
// succeeds because sendCloudLLMPrompt drops all options and retries on a 400
// (see below). Grooming the registry only improves tuning for new models; it
// can never break them. Temperature is deliberately NOT a per-provider
// default — new models reject it — only an explicit choice for known-old
// models.

/** Shared option sets, reused across models that take the same tuning. */
const LOW_TEMPERATURE: ModelOptions = { temperature: 0.1 };
const REASONING_EFFORT_LOW: ModelOptions = { reasoning_effort: 'low' };
const ANTHROPIC_EFFORT_LOW: ModelOptions = { output_config: { effort: 'low' } };
const OPENROUTER_REASONING_LOW: ModelOptions = { reasoning: { effort: 'low' } };

interface ProviderOptionPolicy {
  /** Fully-formed option sets for specific known models, keyed by model id. */
  readonly known: Record<string, ModelOptions>;
  /** Option set used for unknown models of this provider. */
  readonly default: ModelOptions;
}

const OPTION_POLICIES: Partial<Record<AI_PROVIDER_ENUM, ProviderOptionPolicy>> =
  {
    [AI_PROVIDER_ENUM.ANTHROPIC]: {
      // `temperature` is accepted through Opus 4.6 / Sonnet 4.6 (and older) but
      // REMOVED (400) on Opus 4.7+, Opus 4.8, Opus 5, Sonnet 5, and Fable 5 —
      // those steer with `output_config.effort` instead. Note `effort` in turn
      // ERRORS on Sonnet 4.5 / Haiku 4.5, so those must use temperature. Each
      // model gets the determinism lever it actually supports.
      known: {
        // Temperature-supporting models.
        'claude-haiku-4-5-20251001': LOW_TEMPERATURE,
        'claude-opus-4-5-20251101': LOW_TEMPERATURE,
        'claude-opus-4-6': LOW_TEMPERATURE,
        'claude-sonnet-4-5-20250929': LOW_TEMPERATURE,
        'claude-sonnet-4-6': LOW_TEMPERATURE,
        'claude-sonnet-4-20250514': LOW_TEMPERATURE,
        'claude-3-5-sonnet-latest': LOW_TEMPERATURE,
        // Newest generation — temperature removed; steer with low effort.
        'claude-fable-5': ANTHROPIC_EFFORT_LOW,
        'claude-opus-4-7': ANTHROPIC_EFFORT_LOW,
        'claude-opus-4-8': ANTHROPIC_EFFORT_LOW,
        'claude-opus-5': ANTHROPIC_EFFORT_LOW,
        'claude-sonnet-5': ANTHROPIC_EFFORT_LOW,
      },
      // Unknown Anthropic models are almost always newest-gen → low effort,
      // never temperature. A model that rejects effort too falls back to bare.
      default: ANTHROPIC_EFFORT_LOW,
    },
    [AI_PROVIDER_ENUM.OPENAI]: {
      known: {
        'gpt-4o': LOW_TEMPERATURE,
        'gpt-4o-mini': LOW_TEMPERATURE,
        'gpt-4.1': LOW_TEMPERATURE,
      },
      default: REASONING_EFFORT_LOW,
    },
    [AI_PROVIDER_ENUM.GOOGLE]: {
      // Gemini is reached through the OpenAI-compatible endpoint, so it uses
      // `reasoning_effort`. NOTE: verify the accepted values against a live
      // Gemini model — Google's compat mapping has shifted across releases.
      known: {
        'gemini-2.0-flash': LOW_TEMPERATURE,
        'gemini-1.5-flash': LOW_TEMPERATURE,
      },
      default: REASONING_EFFORT_LOW,
    },
    [AI_PROVIDER_ENUM.MISTRAL]: {
      // Mistral has no common reasoning-effort knob; known models take
      // temperature, and unknown models get nothing (never default temperature).
      known: {},
      default: {},
    },
    [AI_PROVIDER_ENUM.OPENROUTER]: {
      // OpenRouter normalises effort across providers via a unified `reasoning`
      // object, so one default works for any routed model.
      known: {},
      default: OPENROUTER_REASONING_LOW,
    },
    [AI_PROVIDER_ENUM.CUSTOM]: {
      // Unknown endpoint and models — send nothing extra.
      known: {},
      default: {},
    },
  };

/** provider+model combinations that rejected our option set (session-scoped). */
const optionsRejectedModels = new Set<string>();

/** Cache key identifying a provider+model combination. */
function providerModelKey(config: CloudLLMConfig): string {
  return `${config.provider}:${config.model}`;
}

/** Resolve the tuning options to send for this provider+model. */
function resolveModelOptions(config: CloudLLMConfig): ModelOptions {
  if (optionsRejectedModels.has(providerModelKey(config))) {
    return {};
  }
  const policy = OPTION_POLICIES[config.provider];
  if (!policy) {
    return {};
  }
  return policy.known[config.model] ?? policy.default;
}

/** Clears the rejected-options cache. Test-only. */
export function resetModelOptionsCache(): void {
  optionsRejectedModels.clear();
}

/** HTTP error carrying the status and raw body for post-hoc inspection. */
class CloudLLMHttpError extends Error {
  constructor(
    readonly status: number,
    readonly body: string,
  ) {
    super(`Cloud LLM API error (${status}): ${body}`);
    this.name = 'CloudLLMHttpError';
  }
}

// ── Public API ───────────────────────────────────────────────────────

/** Perform a single request attempt: fetch, timeout, and error handling. */
async function sendOnce(
  protocol: ProviderProtocol,
  model: string,
  messages: CloudLLMMessage[],
  options: ModelOptions,
): Promise<CloudLLMResponse> {
  const requestBody = protocol.buildRequestBody(model, messages, options);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(protocol.url, {
      method: 'POST',
      headers: protocol.headers,
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      const errorBody = await response.text().catch(() => '');
      throw new CloudLLMHttpError(response.status, errorBody);
    }

    const body: unknown = await response.json();
    return protocol.parseResponse(body);
  } catch (error) {
    clearTimeout(timeout);

    if (error.name === 'AbortError') {
      throw new Error('Cloud LLM request timed out after 30 seconds');
    }

    throw error;
  }
}

/**
 * Send a prompt to a cloud LLM provider and return the response.
 *
 * Sends the model's tuning options (from the registry above). If the model
 * rejects them with an HTTP 400, drops ALL options, retries once with a bare
 * request, and remembers the provider+model so later requests skip options —
 * no error-message parsing, no per-parameter guessing, at most one retry.
 */
export async function sendCloudLLMPrompt(
  config: CloudLLMConfig,
  messages: CloudLLMMessage[],
): Promise<CloudLLMResponse> {
  const protocol = createProtocol(config);
  const options = resolveModelOptions(config);

  try {
    return await sendOnce(protocol, config.model, messages, options);
  } catch (error) {
    const sentOptions = Object.keys(options).length > 0;
    if (
      sentOptions &&
      error instanceof CloudLLMHttpError &&
      error.status === 400
    ) {
      optionsRejectedModels.add(providerModelKey(config));
      return await sendOnce(protocol, config.model, messages, {});
    }
    throw error;
  }
}
