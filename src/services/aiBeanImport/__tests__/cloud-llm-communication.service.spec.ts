import { AI_PROVIDER_ENUM } from '../../../enums/settings/aiProvider';
import {
  CloudLLMConfig,
  CloudLLMMessage,
  resetModelOptionsCache,
  sendCloudLLMPrompt,
} from '../cloud-llm-communication.service';

describe('cloud-llm-communication.service', () => {
  let fetchSpy: jasmine.Spy;

  const systemMessage: CloudLLMMessage = {
    role: 'system',
    content: 'You are a helpful assistant.',
  };
  const userMessage: CloudLLMMessage = {
    role: 'user',
    content: 'Extract bean info from this label.',
  };
  const messages: CloudLLMMessage[] = [systemMessage, userMessage];

  function createConfig(
    overrides: Partial<CloudLLMConfig> = {},
  ): CloudLLMConfig {
    return {
      provider: AI_PROVIDER_ENUM.OPENAI,
      apiKey: 'test-api-key',
      model: 'gpt-4o',
      ...overrides,
    };
  }

  function mockFetchResponse(body: object, status = 200): Response {
    return {
      ok: status >= 200 && status < 300,
      status,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    } as unknown as Response;
  }

  beforeEach(() => {
    fetchSpy = spyOn(globalThis, 'fetch');
    // The rejected-options cache is module-level state that persists across
    // tests; reset it so combinations remembered by one test do not leak.
    resetModelOptionsCache();
  });

  // ── Request building per provider ──────────────────────────────────

  describe('request building', () => {
    // WHY: Each provider has a different base URL, header format, and body shape.
    // These tests verify the correct request is sent for each provider.

    it('should build correct request for OpenAI', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.OPENAI,
      });
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            choices: [{ message: { content: 'response' } }],
            model: 'gpt-4o',
          }),
        ),
      );

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert
      const [url, options] = fetchSpy.calls.mostRecent().args;
      expect(url).toBe('https://api.openai.com/v1/chat/completions');
      expect(options.headers.Authorization).toBe('Bearer test-api-key');
      expect(options.headers['Content-Type']).toBe('application/json');

      const body = JSON.parse(options.body);
      expect(body.model).toBe('gpt-4o');
      expect(body.temperature).toBe(0.1);
      expect(body.messages.length).toBe(2);
      expect(body.messages[0].role).toBe('system');
      expect(body.messages[1].role).toBe('user');
    });

    it('should build correct request for Google (Gemini via OpenAI-compatible endpoint)', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.GOOGLE,
        model: 'gemini-2.0-flash',
      });
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            choices: [{ message: { content: 'response' } }],
            model: 'gemini-2.0-flash',
          }),
        ),
      );

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert
      const [url, options] = fetchSpy.calls.mostRecent().args;
      expect(url).toBe(
        'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
      );
      expect(options.headers.Authorization).toBe('Bearer test-api-key');

      const body = JSON.parse(options.body);
      expect(body.model).toBe('gemini-2.0-flash');
    });

    it('should build correct request for Anthropic', async () => {
      // WHY: Anthropic uses a different auth header, API version header,
      // separate system param, and /messages endpoint.

      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.ANTHROPIC,
        model: 'claude-sonnet-4-20250514',
      });
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            content: [{ text: 'response' }],
            model: 'claude-sonnet-4-20250514',
          }),
        ),
      );

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert
      const [url, options] = fetchSpy.calls.mostRecent().args;
      expect(url).toBe('https://api.anthropic.com/v1/messages');
      expect(options.headers['x-api-key']).toBe('test-api-key');
      expect(options.headers['anthropic-version']).toBe('2023-06-01');
      expect(options.headers.Authorization).toBeUndefined();

      const body = JSON.parse(options.body);
      expect(body.model).toBe('claude-sonnet-4-20250514');
      expect(body.max_tokens).toBe(4096);
      expect(body.temperature).toBe(0.1);
      expect(body.system).toBe('You are a helpful assistant.');
      expect(body.messages.length).toBe(1);
      expect(body.messages[0].role).toBe('user');
    });

    it('should build correct request for OpenRouter', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.OPENROUTER,
        model: 'anthropic/claude-sonnet-4-20250514',
      });
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            choices: [{ message: { content: 'response' } }],
            model: 'anthropic/claude-sonnet-4-20250514',
          }),
        ),
      );

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert
      const [url, options] = fetchSpy.calls.mostRecent().args;
      expect(url).toBe('https://openrouter.ai/api/v1/chat/completions');
      expect(options.headers.Authorization).toBe('Bearer test-api-key');
      expect(options.headers['HTTP-Referer']).toBe('https://beanconqueror.com');
      expect(options.headers['X-OpenRouter-Title']).toBe('Beanconqueror');

      const body = JSON.parse(options.body);
      expect(body.model).toBe('anthropic/claude-sonnet-4-20250514');
    });

    it('should strip trailing slash from custom base URL', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.CUSTOM,
        model: 'my-model',
        baseUrl: 'https://my-llm.example.com/api/',
      });
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            choices: [{ message: { content: 'response' } }],
            model: 'my-model',
          }),
        ),
      );

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert
      const [url] = fetchSpy.calls.mostRecent().args;
      expect(url).toBe('https://my-llm.example.com/api/chat/completions');
    });

    it('should send Anthropic body without system message when none provided', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.ANTHROPIC,
        model: 'claude-sonnet-4-20250514',
      });
      const userOnly: CloudLLMMessage[] = [userMessage];
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            content: [{ text: 'response' }],
            model: 'claude-sonnet-4-20250514',
          }),
        ),
      );

      // Act
      await sendCloudLLMPrompt(config, userOnly);

      // Assert
      const body = JSON.parse(fetchSpy.calls.mostRecent().args[1].body);
      expect(body.system).toBe('');
      expect(body.messages.length).toBe(1);
    });
  });

  // ── Response parsing ───────────────────────────────────────────────

  describe('response parsing', () => {
    it('should parse OpenAI-compatible response with usage', async () => {
      // Arrange
      const config = createConfig();
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            choices: [{ message: { content: 'Parsed bean info' } }],
            model: 'gpt-4o',
            usage: { prompt_tokens: 100, completion_tokens: 50 },
          }),
        ),
      );

      // Act
      const result = await sendCloudLLMPrompt(config, messages);

      // Assert
      expect(result.content).toBe('Parsed bean info');
      expect(result.model).toBe('gpt-4o');
      expect(result.usage).toEqual({
        prompt_tokens: 100,
        completion_tokens: 50,
      });
    });

    it('should parse OpenAI-compatible response without usage', async () => {
      // Arrange
      const config = createConfig();
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            choices: [{ message: { content: 'response' } }],
            model: 'gpt-4o',
          }),
        ),
      );

      // Act
      const result = await sendCloudLLMPrompt(config, messages);

      // Assert
      expect(result.content).toBe('response');
      expect(result.usage).toBeUndefined();
    });

    it('should parse Anthropic response format', async () => {
      // WHY: Anthropic uses content[].text instead of choices[].message.content,
      // and input_tokens/output_tokens instead of prompt_tokens/completion_tokens.

      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.ANTHROPIC,
        model: 'claude-sonnet-4-20250514',
      });
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            content: [{ text: 'Anthropic response' }],
            model: 'claude-sonnet-4-20250514',
            usage: { input_tokens: 80, output_tokens: 40 },
          }),
        ),
      );

      // Act
      const result = await sendCloudLLMPrompt(config, messages);

      // Assert
      expect(result.content).toBe('Anthropic response');
      expect(result.model).toBe('claude-sonnet-4-20250514');
      expect(result.usage).toEqual({
        prompt_tokens: 80,
        completion_tokens: 40,
      });
    });

    it('should handle empty/missing content gracefully', async () => {
      // WHY: API may return empty choices array or missing fields

      // Arrange
      const config = createConfig();
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            choices: [],
            model: 'gpt-4o',
          }),
        ),
      );

      // Act
      const result = await sendCloudLLMPrompt(config, messages);

      // Assert
      expect(result.content).toBe('');
      expect(result.model).toBe('gpt-4o');
    });

    it('should handle empty Anthropic content array gracefully', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.ANTHROPIC,
        model: 'claude-sonnet-4-20250514',
      });
      fetchSpy.and.returnValue(
        Promise.resolve(
          mockFetchResponse({
            content: [],
            model: 'claude-sonnet-4-20250514',
          }),
        ),
      );

      // Act
      const result = await sendCloudLLMPrompt(config, messages);

      // Assert
      expect(result.content).toBe('');
    });
  });

  // ── Error handling ─────────────────────────────────────────────────

  describe('error handling', () => {
    it('should throw descriptive error on HTTP 401 (unauthorized)', async () => {
      // Arrange
      const config = createConfig();
      fetchSpy.and.returnValue(
        Promise.resolve({
          ok: false,
          status: 401,
          text: () => Promise.resolve('Invalid API key'),
        } as unknown as Response),
      );

      // Act & Assert
      await expectAsync(
        sendCloudLLMPrompt(config, messages),
      ).toBeRejectedWithError('Cloud LLM API error (401): Invalid API key');
    });

    it('should handle error response when body text extraction fails', async () => {
      // WHY: response.text() itself might fail; we should still get a useful error

      // Arrange
      const config = createConfig();
      fetchSpy.and.returnValue(
        Promise.resolve({
          ok: false,
          status: 403,
          text: () => Promise.reject(new Error('stream error')),
        } as unknown as Response),
      );

      // Act & Assert
      await expectAsync(
        sendCloudLLMPrompt(config, messages),
      ).toBeRejectedWithError('Cloud LLM API error (403): ');
    });

    it('should throw timeout error when request exceeds 30 seconds', async () => {
      // WHY: We use AbortController to enforce a 30s timeout

      // Arrange
      const config = createConfig();
      const abortError = new DOMException(
        'The operation was aborted',
        'AbortError',
      );
      fetchSpy.and.returnValue(Promise.reject(abortError));

      // Act & Assert
      await expectAsync(
        sendCloudLLMPrompt(config, messages),
      ).toBeRejectedWithError('Cloud LLM request timed out after 30 seconds');
    });

    it('should re-throw network errors as-is', async () => {
      // Arrange
      const config = createConfig();
      fetchSpy.and.returnValue(
        Promise.reject(new TypeError('Failed to fetch')),
      );

      // Act & Assert
      await expectAsync(sendCloudLLMPrompt(config, messages)).toBeRejectedWith(
        jasmine.any(TypeError),
      );
    });
  });

  // ── Model-options registry ─────────────────────────────────────────

  describe('model-options registry', () => {
    // WHY: Each provider+model gets a fully-formed option set (low temperature
    // for known-old models, low reasoning effort for newer ones). Unknown
    // models get a per-provider default. On a 400, all options are dropped and
    // the request is retried bare, then remembered.

    const success = (model: string) =>
      mockFetchResponse({
        choices: [{ message: { content: 'response' } }],
        model,
      });

    const badRequest = () =>
      mockFetchResponse({ error: { message: 'unsupported parameter' } }, 400);

    it('should send a low reasoning effort for an unknown OpenAI model', async () => {
      // Arrange
      const config = createConfig({ model: 'gpt-5.6-terra' });
      fetchSpy.and.returnValue(Promise.resolve(success('gpt-5.6-terra')));

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert: provider default, not temperature.
      const body = JSON.parse(fetchSpy.calls.mostRecent().args[1].body);
      expect(body.reasoning_effort).toBe('low');
      expect(body.temperature).toBeUndefined();
    });

    const anthropicSuccess = (model: string) =>
      mockFetchResponse({ content: [{ text: 'response' }], model });

    const anthropicBody = async (model: string) => {
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.ANTHROPIC,
        model,
      });
      fetchSpy.and.returnValue(Promise.resolve(anthropicSuccess(model)));
      await sendCloudLLMPrompt(config, messages);
      return JSON.parse(fetchSpy.calls.mostRecent().args[1].body);
    };

    it('should send low effort for newest-gen Anthropic models (temperature removed)', async () => {
      for (const model of [
        'claude-sonnet-5',
        'claude-opus-4-8',
        'claude-fable-5',
      ]) {
        const body = await anthropicBody(model);
        expect(body.output_config).toEqual({ effort: 'low' });
        expect(body.temperature).toBeUndefined();
      }
    });

    it('should send low temperature (not effort) for models that still accept it', async () => {
      // Regression: Sonnet 4.6 / Opus 4.6 accept temperature — must not be effort-only.
      for (const model of ['claude-sonnet-4-6', 'claude-opus-4-6']) {
        const body = await anthropicBody(model);
        expect(body.temperature).toBe(0.1);
        expect(body.output_config).toBeUndefined();
      }
    });

    it('should send temperature (never effort) for Haiku 4.5 / Sonnet 4.5, where effort errors', async () => {
      for (const model of [
        'claude-haiku-4-5-20251001',
        'claude-sonnet-4-5-20250929',
      ]) {
        const body = await anthropicBody(model);
        expect(body.temperature).toBe(0.1);
        expect(body.output_config).toBeUndefined();
        expect(body.reasoning_effort).toBeUndefined();
      }
    });

    it('should send low effort for an unknown Anthropic model (newest-gen bet)', async () => {
      const body = await anthropicBody('claude-opus-9');
      expect(body.output_config).toEqual({ effort: 'low' });
      expect(body.temperature).toBeUndefined();
    });

    it('should send the unified reasoning option for unknown OpenRouter models', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.OPENROUTER,
        model: 'x-ai/grok-5',
      });
      fetchSpy.and.returnValue(Promise.resolve(success('x-ai/grok-5')));

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert
      const body = JSON.parse(fetchSpy.calls.mostRecent().args[1].body);
      expect(body.reasoning).toEqual({ effort: 'low' });
    });

    it('should send no extra options for an unknown Custom-provider model', async () => {
      // Arrange
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.CUSTOM,
        model: 'my-model',
        baseUrl: 'https://my-llm.example.com',
      });
      fetchSpy.and.returnValue(Promise.resolve(success('my-model')));

      // Act
      await sendCloudLLMPrompt(config, messages);

      // Assert
      const body = JSON.parse(fetchSpy.calls.mostRecent().args[1].body);
      expect(body.temperature).toBeUndefined();
      expect(body.reasoning_effort).toBeUndefined();
    });

    it('should drop all options and retry bare when the model rejects them with a 400', async () => {
      // Arrange
      const config = createConfig({ model: 'gpt-5.6-terra' });
      fetchSpy.and.returnValues(
        Promise.resolve(badRequest()),
        Promise.resolve(success('gpt-5.6-terra')),
      );

      // Act
      const result = await sendCloudLLMPrompt(config, messages);

      // Assert
      expect(result.content).toBe('response');
      expect(fetchSpy).toHaveBeenCalledTimes(2);

      const firstBody = JSON.parse(fetchSpy.calls.argsFor(0)[1].body);
      const retryBody = JSON.parse(fetchSpy.calls.argsFor(1)[1].body);
      expect(firstBody.reasoning_effort).toBe('low');
      expect(retryBody.reasoning_effort).toBeUndefined();
    });

    it('should remember a rejected model and send bare on later requests', async () => {
      // Arrange: first request rejects then succeeds, populating the cache.
      const config = createConfig({ model: 'gpt-5.6-terra' });
      fetchSpy.and.returnValues(
        Promise.resolve(badRequest()),
        Promise.resolve(success('gpt-5.6-terra')),
      );
      await sendCloudLLMPrompt(config, messages);

      // Act: a second request to the same model.
      fetchSpy.calls.reset();
      fetchSpy.and.returnValue(Promise.resolve(success('gpt-5.6-terra')));
      await sendCloudLLMPrompt(config, messages);

      // Assert: bare on the first attempt, no retry needed.
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const body = JSON.parse(fetchSpy.calls.argsFor(0)[1].body);
      expect(body.reasoning_effort).toBeUndefined();
    });

    it('should not retry when there were no options to drop (Custom provider 400)', async () => {
      // Arrange: Custom provider sends no options, so a 400 is a real error.
      const config = createConfig({
        provider: AI_PROVIDER_ENUM.CUSTOM,
        model: 'my-model',
        baseUrl: 'https://my-llm.example.com',
      });
      fetchSpy.and.returnValue(Promise.resolve(badRequest()));

      // Act & Assert
      await expectAsync(sendCloudLLMPrompt(config, messages)).toBeRejected();
      expect(fetchSpy).toHaveBeenCalledTimes(1);
    });
  });
});
