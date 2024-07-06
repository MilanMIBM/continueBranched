import { BaseLLM } from "..";
import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  ModelProvider,
} from "../..";
import { streamResponse, streamSse } from "../stream";

class IBMGenerativeAI extends BaseLLM {
  static providerName: ModelProvider = "ibm-generative-ai";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://bam-api.res.ibm.com/v2",
    model: "mixstral-8x7b-instruct",
  };
  private _apiDateVersion = "2024-01-10";

  private _getModel() {
    return (
      {
        "falcon-40b": "tiiuae/falcon-40b",
        "flan-ul2": "google/flan-ul2",
        "mistral-tiny": "mistralai/mistral-7b-instruct-v0-2",
        "mixstral-8x7b-instruct": "mistralai/mixtral-8x7b-instruct-v0-1",
        "sqlcoder-34b-alpha": "defog/sqlcoder-34b-alpha",
        "granite-13b-chat": "ibm/granite-13b-chat-v2",
        "granite-13b-instruct": "ibm/granite-13b-instruct-v2",
        "granite-20b-code-instruct-v1": "ibm/granite-20b-code-instruct-v1",
        "granite-20b-code-instruct-v1-gptq":
          "ibm/granite-20b-code-instruct-v1-gptq",
        "llama2-7b": "meta-llama/llama-2-7b-chat",
        "llama2-13b": "meta-llama/llama-2-13b-chat",
        "llama2-70b": "meta-llama/llama-2-70b-chat",
        "codellama-34b": "codellama/codellama-34b-instruct",
        "codellama-70b": "codellama/codellama-70b-instruct",
      }[this.model] || this.model
    );
  }

  private _convertArgs(
    options: CompletionOptions,
    prompt: string | ChatMessage[]
  ) {
    let parameters = {};
    if (options.decodingMethod === "greedy") {
      parameters = {
        decoding_method: options.decodingMethod,
        max_new_tokens: options.maxTokens,
        min_new_tokens: options.minTokens,
        stop_sequences: options.stop,
        include_stop_sequence: options.includeStopSequence,
        repetition_penalty: options.repetitionPenalty,
      };
    } else if (options.decodingMethod === "sample") {
      parameters = {
        decoding_method: options.decodingMethod,
        temperature: options.temperature,
        top_p: options.topP,
        top_k: options.topK,
        // typical_p: 1,
        // random_seed: 3,
        repetition_penalty: options.repetitionPenalty,
        stop_sequences: options.stop,
        include_stop_sequence: options.includeStopSequence,
        min_new_tokens: options.minTokens,
        max_new_tokens: options.maxTokens,
      };
    }

    const finalOptions: any = {
      model_id: this._getModel(),
      parameters,
      moderations: {
        hap: { threshold: 0.75, input: true, output: true },
        stigma: { threshold: 0.75, input: true, output: true },
      },
    };

    if (typeof prompt === "string") {
      finalOptions.input = `${prompt}
`;
    } else {
      finalOptions.messages = prompt;
    }

    console.log("finalOptions:", JSON.stringify(finalOptions));

    return finalOptions;
  }

  private _body(options: CompletionOptions, prompt: string | ChatMessage[]) {
    return {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(this._convertArgs(options, prompt)),
    };
  }

  protected async _complete(
    prompt: string,
    options: CompletionOptions
  ): Promise<string> {
    const response = await this.fetch(
      `${this.apiBase}/text/generation?version=${this._apiDateVersion}`,
      this._body(options, prompt)
    );

    const result = await response.json();
    return result.completion;
  }

  protected async *_streamComplete(
    prompt: string,
    options: CompletionOptions
  ): AsyncGenerator<string> {
    const response = await this.fetch(
      `${this.apiBase}/text/generation_stream?version=${this._apiDateVersion}`,
      this._body(options, prompt)
    );

    for await (const value of streamSse(response)) {
      if (value.results) {
        yield value.results[0].generated_text;
      }
    }
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    options: CompletionOptions
  ): AsyncGenerator<ChatMessage> {
    const response = await this.fetch(
      `${this.apiBase}/text/chat_stream?version=${this._apiDateVersion}`,
      this._body(options, messages)
    );

    let buffer = "";
    for await (const value of streamResponse(response)) {
      // Append the received chunk to the buffer

      buffer += value;
      // Split the buffer into individual JSON chunks
      const chunks = buffer.split("\n");
      buffer = chunks.pop() ?? "";
      for (let chunk of chunks) {
        if (chunk.trim() !== "") {
          try {
            let j;

            if (!chunk.startsWith("data:")) {
              continue;
            }
            j = JSON.parse(chunk.substring(6));

            if (j.results?.[0].generated_text) {
              yield {
                role: "assistant",
                content: j.results[0].generated_text,
              };
            } else if (j.error) {
              throw new Error(j.error);
            }
          } catch (e) {
            throw new Error(
              `Error parsing IBM Generative AI response: ${e} ${chunk}`
            );
          }
        }
      }
    }
  }
}

export default IBMGenerativeAI;