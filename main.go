// Command agent is an interactive CLI assistant backed by MCP Toolbox.
//
// It discovers whatever tools the Toolbox server exposes at startup and hands
// them to the model, so adding a capability means editing the Toolbox YAML —
// no queries are hard-coded here.
//
// Usage:
//
//	docker compose up -d
//	./toolbox/toolbox --config toolbox/food-tools.yaml
//	go run . [-toolbox URL] [-toolset NAME[,NAME...]] [-model NAME]
package main

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/googleapis/mcp-toolbox-sdk-go/core"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

// systemPrompt describes the assistant's job to the model. It lives in its own
// file so it can be edited as prose, and is compiled into the binary so the
// agent still runs from any working directory.
//
//go:embed prompts/system.md
var systemPrompt string

const replHelp = `Commands:
  /help    show this help
  /tools   list the tools loaded from Toolbox
  /reset   forget the conversation so far
  /exit    quit (Ctrl-D works too)
`

// maxToolRounds bounds how many times the model may call tools while answering
// a single question. A model that keeps calling tools without ever producing a
// reply would otherwise spin forever; on the last round the tools are withheld
// so it has to answer in prose.
const maxToolRounds = 8

// config holds everything the agent needs to reach its dependencies. Each
// field is a flag with an environment-variable fallback, so the same binary
// serves any Toolbox instance and toolset.
type config struct {
	toolboxURL string
	toolsets   []string
	model      string
	ollamaURL  string
}

func parseConfig() config {
	var (
		// Use 127.0.0.1, not "localhost": on macOS the AirPlay Receiver
		// (ControlCenter) also listens on *:5000 over IPv6, and "localhost"
		// resolves to ::1 first, so requests land on AirTunes and get a bare
		// 403. Toolbox binds IPv4 127.0.0.1 by default.
		toolboxURL = flag.String("toolbox", envOr("TOOLBOX_URL", "http://127.0.0.1:5000"),
			"base URL of the MCP Toolbox server")
		toolsets = flag.String("toolset", os.Getenv("TOOLBOX_TOOLSET"),
			"comma-separated toolsets to load; empty loads every tool the server exposes")
		model = flag.String("model", envOr("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
			"model name to run")
		ollamaURL = flag.String("ollama", envOr("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
			"OpenAI-compatible endpoint serving the model")
	)
	flag.Parse()

	return config{
		toolboxURL: *toolboxURL,
		toolsets:   splitList(*toolsets),
		model:      *model,
		ollamaURL:  *ollamaURL,
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// splitList turns "a, b" into ["a" "b"]. An empty input yields a single empty
// name, which the Toolbox SDK reads as "the default toolset" — every tool the
// server has loaded.
func splitList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

// convertToLangchainTool converts a generic core.ToolboxTool into a
// LangChainGo llms.Tool.
func convertToLangchainTool(toolboxTool *core.ToolboxTool) llms.Tool {
	// Fetch the tool's input schema
	inputschema, err := toolboxTool.InputSchema()
	if err != nil {
		return llms.Tool{}
	}

	var paramsSchema map[string]any
	_ = json.Unmarshal(inputschema, &paramsSchema)

	// Convert into LangChain's llms.Tool
	return llms.Tool{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        toolboxTool.Name(),
			Description: toolboxTool.Description(),
			Parameters:  paramsSchema,
		},
	}
}

// loadTools loads every requested toolset, dropping duplicates so overlapping
// toolsets don't hand the model the same tool twice.
func loadTools(ctx context.Context, client *core.ToolboxClient, toolsets []string) ([]*core.ToolboxTool, error) {
	var tools []*core.ToolboxTool
	seen := make(map[string]bool)

	for _, name := range toolsets {
		loaded, err := client.LoadToolset(name, ctx)
		if err != nil {
			return nil, err
		}
		for _, tool := range loaded {
			if seen[tool.Name()] {
				continue
			}
			seen[tool.Name()] = true
			tools = append(tools, tool)
		}
	}
	return tools, nil
}

// printAnswer prints the assistant's reply in a bordered, readable block.
// The query isn't echoed back — the user just typed it.
func printAnswer(answer string) {
	const width = 78
	answer = strings.TrimSpace(answer)
	if answer == "" {
		answer = "(no reply)"
	}
	fmt.Println("╭" + strings.Repeat("─", width) + "╮")
	fmt.Printf("│ %s\n", strings.ReplaceAll(answer, "\n", "\n│ "))
	fmt.Println("╰" + strings.Repeat("─", width) + "╯")
}

// ask runs a single user turn: send the query, run whatever tools the model
// asks for, and repeat until it replies with prose instead of another tool
// call.
//
// The history is returned rather than mutated in place, because on failure it
// is rolled back to its pre-turn state. A turn that dies midway would
// otherwise leave an assistant message with tool_calls that have no matching
// tool responses, which the API rejects on every subsequent request — one
// transient error would poison the rest of the session.
func ask(
	ctx context.Context,
	llm llms.Model,
	history []llms.MessageContent,
	langchainTools []llms.Tool,
	toolsMap map[string]*core.ToolboxTool,
	query string,
) ([]llms.MessageContent, string, error) {
	snapshot := len(history)
	fail := func(err error) ([]llms.MessageContent, string, error) {
		return history[:snapshot], "", err
	}

	history = append(history, llms.TextParts(llms.ChatMessageTypeHuman, query))

	for round := 0; ; round++ {
		// Offer the tools on every round but the last, where the model has
		// used up its budget and must summarize what it already has.
		var opts []llms.CallOption
		if round < maxToolRounds {
			opts = append(opts, llms.WithTools(langchainTools))
		}

		resp, err := llm.GenerateContent(ctx, history, opts...)
		if err != nil {
			return fail(fmt.Errorf("LLM call failed: %w", err))
		}
		respChoice := resp.Choices[0]

		assistantResponse := llms.TextParts(llms.ChatMessageTypeAI, respChoice.Content)
		for _, tc := range respChoice.ToolCalls {
			assistantResponse.Parts = append(assistantResponse.Parts, tc)
		}
		history = append(history, assistantResponse)

		if len(respChoice.ToolCalls) == 0 {
			return history, respChoice.Content, nil
		}

		// Process each tool call requested by the model.
		for _, tc := range respChoice.ToolCalls {
			toolName := tc.FunctionCall.Name
			tool, ok := toolsMap[toolName]
			if !ok {
				return fail(fmt.Errorf("model called unknown tool %q", toolName))
			}
			fmt.Printf("  ↳ %s%s\n", toolName, tc.FunctionCall.Arguments)

			var args map[string]any
			if err := json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args); err != nil {
				return fail(fmt.Errorf("failed to unmarshal arguments for tool %q: %w", toolName, err))
			}
			toolResult, err := tool.Invoke(ctx, args)
			if err != nil {
				return fail(fmt.Errorf("failed to execute tool %q: %w", toolName, err))
			}
			if toolResult == "" || toolResult == nil {
				toolResult = "Operation completed successfully with no specific return value."
			}

			// Create the tool call response message and add it to the history.
			// ToolCallID must match the id the model issued the call with, so
			// the model can correlate this result back to that specific call.
			history = append(history, llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{
					llms.ToolCallResponse{
						ToolCallID: tc.ID,
						Name:       toolName,
						Content:    fmt.Sprintf("%v", toolResult),
					},
				},
			})
		}
	}
}

func main() {
	cfg := parseConfig()
	ctx := context.Background()

	// Initialize the LLM client against Ollama's OpenAI-compatible endpoint.
	// Requires a local Ollama server (https://ollama.com) running with the
	// model pulled, e.g.: ollama pull llama3.1
	// The langchaingo native "ollama" backend doesn't forward tool
	// definitions or parse tool_calls out of the response, so tool calling
	// silently never happens there. The OpenAI-compatible endpoint does both.
	llm, err := openai.New(
		openai.WithBaseURL(cfg.ollamaURL),
		openai.WithToken("ollama"),
		openai.WithModel(cfg.model),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	// Initialize the MCP Toolbox client.
	toolboxClient, err := core.NewToolboxClient(cfg.toolboxURL)
	if err != nil {
		log.Fatalf("Failed to create Toolbox client: %v", err)
	}

	tools, err := loadTools(ctx, toolboxClient, cfg.toolsets)
	if err != nil {
		log.Fatalf("Failed to load tools: %v\nMake sure the Toolbox server is running at %s with your tool configs loaded.", err, cfg.toolboxURL)
	}
	if len(tools) == 0 {
		log.Fatalf("Toolbox at %s exposed no tools; check its configuration.", cfg.toolboxURL)
	}

	toolsMap := make(map[string]*core.ToolboxTool, len(tools))
	langchainTools := make([]llms.Tool, len(tools))
	// Convert the loaded ToolboxTools into the format LangChainGo requires.
	for i, tool := range tools {
		langchainTools[i] = convertToLangchainTool(tool)
		toolsMap[tool.Name()] = tool
	}

	// Start the conversation history.
	messageHistory := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, systemPrompt),
	}

	fmt.Printf("Agent ready — %d tools from %s, model %s\n", len(tools), cfg.toolboxURL, cfg.model)
	fmt.Println("Ask a question, or /help for commands.")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nyou> ")
		if !scanner.Scan() {
			// EOF (Ctrl-D) or a read error; either way the session is over.
			fmt.Println()
			break
		}

		query := strings.TrimSpace(scanner.Text())
		switch query {
		case "":
			continue
		case "/exit", "/quit":
			return
		case "/help":
			fmt.Print(replHelp)
			continue
		case "/tools":
			for _, tool := range tools {
				fmt.Printf("  %-28s %s\n", tool.Name(), tool.Description())
			}
			continue
		case "/reset":
			// Keep the system prompt, drop everything said since.
			messageHistory = messageHistory[:1]
			fmt.Println("Conversation history cleared.")
			continue
		}

		// Errors here are per-turn, not fatal: a bad model reply or a failed
		// query shouldn't end an interactive session.
		newHistory, answer, err := ask(ctx, llm, messageHistory, langchainTools, toolsMap, query)
		messageHistory = newHistory
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			continue
		}
		printAnswer(answer)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Failed to read input: %v", err)
	}
}
