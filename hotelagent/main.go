package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/googleapis/mcp-toolbox-sdk-go/core"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

// ConvertToLangchainTool converts a generic core.ToolboxTool into a LangChainGo llms.Tool.
func ConvertToLangchainTool(toolboxTool *core.ToolboxTool) llms.Tool {

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

const systemPrompt = `
You're a helpful hotel assistant. You handle hotel searching, booking, and
cancellations. When the user searches for a hotel, mention its name, id,
location and price tier. Always mention hotel ids while performing any
searches. This is very important for any operations. For any bookings or
cancellations, please provide the appropriate confirmation. Be sure to
update checkin or checkout dates if mentioned by the user.
Don't ask for confirmations from the user.
`

// printExchange prints a query/answer pair in a bordered, readable block.
func printExchange(n int, query, answer string) {
	const width = 78
	fmt.Println()
	fmt.Println("╔" + strings.Repeat("═", width) + "╗")
	fmt.Printf("║ Query %-*d║\n", width-7, n)
	fmt.Println("╟" + strings.Repeat("─", width) + "╢")
	fmt.Printf("║ %s\n", query)
	fmt.Println("╟" + strings.Repeat("─", width) + "╢")
	fmt.Println("║ Answer:")
	fmt.Printf("║ %s\n", strings.ReplaceAll(strings.TrimSpace(answer), "\n", "\n║ "))
	fmt.Println("╚" + strings.Repeat("═", width) + "╝")
}

var queries = []string{
	"Find hotels in Basel with Basel in its name.",
	"Can you book the hotel Hilton Basel for me?",
	"Oh wait, this is too expensive. Please cancel it.",
	"Please book the Hyatt Regency instead.",
	"My check in dates would be from April 10, 2024 to April 19, 2024.",
}

func main() {
	toolboxURL := "http://localhost:5000"
	ctx := context.Background()

	// Initialize the LLM client against Ollama's OpenAI-compatible endpoint.
	// Requires a local Ollama server (https://ollama.com) running with the
	// model pulled, e.g.: ollama pull llama3.1
	// The langchaingo native "ollama" backend doesn't forward tool
	// definitions or parse tool_calls out of the response, so tool calling
	// silently never happens there. The OpenAI-compatible endpoint does both.
	ollamaModel := os.Getenv("OLLAMA_MODEL")
	if ollamaModel == "" {
		ollamaModel = "gpt-oss:120b-cloud"
	}
	llm, err := openai.New(
		openai.WithBaseURL("http://localhost:11434/v1"),
		openai.WithToken("ollama"),
		openai.WithModel(ollamaModel),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	// Initialize the MCP Toolbox client.
	toolboxClient, err := core.NewToolboxClient(toolboxURL)
	if err != nil {
		log.Fatalf("Failed to create Toolbox client: %v", err)
	}

	// Load the tool using the MCP Toolbox SDK.
	tools, err := toolboxClient.LoadToolset("my-toolset", ctx)
	if err != nil {
		log.Fatalf("Failed to load tools: %v\nMake sure your Toolbox server is running and the tool is configured.", err)
	}

	toolsMap := make(map[string]*core.ToolboxTool, len(tools))

	langchainTools := make([]llms.Tool, len(tools))
	// Convert the loaded ToolboxTools into the format LangChainGo requires.
	for i, tool := range tools {
		langchainTools[i] = ConvertToLangchainTool(tool)
		toolsMap[tool.Name()] = tool
	}

	// Start the conversation history.
	messageHistory := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, systemPrompt),
	}

	for i, query := range queries {
		messageHistory = append(messageHistory, llms.TextParts(llms.ChatMessageTypeHuman, query))

		// Make the first call to the LLM, making it aware of the tool.
		resp, err := llm.GenerateContent(ctx, messageHistory, llms.WithTools(langchainTools))
		if err != nil {
			log.Fatalf("LLM call failed: %v", err)
		}
		respChoice := resp.Choices[0]

		assistantResponse := llms.TextParts(llms.ChatMessageTypeAI, respChoice.Content)
		for _, tc := range respChoice.ToolCalls {
			assistantResponse.Parts = append(assistantResponse.Parts, tc)
		}
		messageHistory = append(messageHistory, assistantResponse)

		// Process each tool call requested by the model.
		for _, tc := range respChoice.ToolCalls {
			toolName := tc.FunctionCall.Name
			tool := toolsMap[toolName]
			var args map[string]any
			if err := json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args); err != nil {
				log.Fatalf("Failed to unmarshal arguments for tool '%s': %v", toolName, err)
			}
			toolResult, err := tool.Invoke(ctx, args)
			if err != nil {
				log.Fatalf("Failed to execute tool '%s': %v", toolName, err)
			}
			if toolResult == "" || toolResult == nil {
				toolResult = "Operation completed successfully with no specific return value."
			}

			// Create the tool call response message and add it to the history.
			// ToolCallID must match the id the model issued the call with, so
			// the model can correlate this result back to that specific call.
			toolResponse := llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{
					llms.ToolCallResponse{
						ToolCallID: tc.ID,
						Name:       toolName,
						Content:    fmt.Sprintf("%v", toolResult),
					},
				},
			}
			messageHistory = append(messageHistory, toolResponse)
		}

		answer := respChoice.Content
		if len(respChoice.ToolCalls) > 0 {
			// Only make a follow-up call when tools actually ran; the model
			// needs a turn to summarize the tool results into a reply.
			finalResp, err := llm.GenerateContent(ctx, messageHistory)
			if err != nil {
				log.Fatalf("Final LLM call failed after tool execution: %v", err)
			}
			answer = finalResp.Choices[0].Content
			messageHistory = append(messageHistory, llms.TextParts(llms.ChatMessageTypeAI, answer))
		}

		printExchange(i+1, query, answer)
	}

}
