import type { IDE, IndexTag, IndexingProgressUpdate } from "..";
import type { ConfigHandler } from "../config/handler";
import { ContinueServerClient } from "../continueServer/stubs/client";
import { CodeSnippetsCodebaseIndex } from "./CodeSnippetsIndex";
import { FullTextSearchCodebaseIndex } from "./FullTextSearch";
import { LanceDbIndex } from "./LanceDbIndex";
import { ChunkCodebaseIndex } from "./chunk/ChunkCodebaseIndex";
import { getComputeDeleteAddRemove } from "./refreshIndex";
import type { CodebaseIndex } from "./types";

export class PauseToken {
  constructor(private _paused: boolean) {}

  set paused(value: boolean) {
    this._paused = value;
  }

  get paused(): boolean {
    return this._paused;
  }
}

export class CodebaseIndexer {
  private continueServerClient?: ContinueServerClient;
  constructor(
    private readonly configHandler: ConfigHandler,
    private readonly ide: IDE,
    private readonly pauseToken: PauseToken,
    private readonly continueServerUrl: string | undefined,
    private readonly userToken: Promise<string | undefined>,
  ) {
    if (continueServerUrl) {
      this.continueServerClient = new ContinueServerClient(
        continueServerUrl,
        userToken,
      );
    }
  }

  private async getIndexesToBuild(): Promise<CodebaseIndex[]> {
    const config = await this.configHandler.loadConfig();

    const indexes = [
      new ChunkCodebaseIndex(
        this.ide.readFile.bind(this.ide),
        this.continueServerClient,
      ), // Chunking must come first
      new LanceDbIndex(
        config.embeddingsProvider,
        this.ide.readFile.bind(this.ide),
        this.continueServerClient,
      ),
      new FullTextSearchCodebaseIndex(),
      new CodeSnippetsCodebaseIndex(this.ide),
    ];

    return indexes;
  }

  async *refresh(
    workspaceDirs: string[],
    abortSignal: AbortSignal,
  ): AsyncGenerator<IndexingProgressUpdate> {
    if (workspaceDirs.length === 0) {
      yield {
        progress: 0,
        desc: "Nothing to index",
        status: "disabled",     
      };
      return;
    }
    
    const config = await this.configHandler.loadConfig();
    if (config.disableIndexing) {
      yield {
        progress: 0,
        desc: "Indexing is disabled in config.json",
        status: "disabled",
      };
      return;
    } else {
      yield {
        progress: 0,
        desc: "Starting indexing",
        status: "loading",
      };
    }

    const indexesToBuild = await this.getIndexesToBuild();

    let completedDirs = 0;

    // Wait until Git Extension has loaded to report progress
    // so we don't appear stuck at 0% while waiting
    if (workspaceDirs.length > 0) {
      await this.ide.getRepoName(workspaceDirs[0]);
    }
    yield {
      progress: 0,
      desc: "Starting indexing...",
      status: "loading",
    };

    for (const directory of workspaceDirs) {
      const stats = await this.ide.getStats(directory);
      const branch = await this.ide.getBranch(directory);
      const repoName = await this.ide.getRepoName(directory);
      let completedIndexes = 0;

      try {
        for (const codebaseIndex of indexesToBuild) {
          // TODO: IndexTag type should use repoName rather than directory
          const tag: IndexTag = {
            directory,
            branch,
            artifactId: codebaseIndex.artifactId,
          };
          const [results, markComplete] = await getComputeDeleteAddRemove(
            tag,
            { ...stats },
            (filepath) => this.ide.readFile(filepath),
            repoName,
          );

          for await (const { progress, desc } of codebaseIndex.update(
            tag,
            results,
            markComplete,
            repoName,
          )) {
            // Handle pausing in this loop because it's the only one really taking time
            if (abortSignal.aborted) {
              yield {
                progress: 1,
                desc: "Indexing cancelled",
              };
              return;
            }
            while (this.pauseToken.paused) {
              await new Promise((resolve) => setTimeout(resolve, 100));
            }

            yield {
              progress:
                (completedDirs +
                  (completedIndexes + progress) / indexesToBuild.length) /
                workspaceDirs.length,
              desc,
              status: "indexing",
            };
          }
          completedIndexes++;
          yield {
            progress:
              (completedDirs + completedIndexes / indexesToBuild.length) /
              workspaceDirs.length,
            desc: `Completed indexing ${codebaseIndex.artifactId}`,
          };
        } catch (e) {
          yield {
            progress: 0, 
            desc: `${e}`,
            status: "failed"
          }
          console.warn(
            `Error updating the ${codebaseIndex.artifactId} index: ${e}`,
          );
          return
        }
      }

      completedDirs++;
      yield {
        progress: completedDirs / workspaceDirs.length,
        desc: "Indexing Complete",
        status: "done",
      };
    }
  }
}
