# How to Use the Pre-Built ReAct Agent

This guide demonstrates how to create a simple ReAct agent application that can check the weather. The application consists of an agent (LLM) and tools, allowing for interactive user queries.

## Prerequisites

This guide assumes familiarity with the following concepts:

- [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)
- [Tools](https://python.langchain.com/docs/concepts/tools/)

## Overview

In this tutorial, we will create a ReAct agent that can respond to weather inquiries. The agent will decide whether to use tools based on user input.

> **Note:** We will use a [prebuilt agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent). While this is a great way to get started quickly, we recommend learning how to build your own agent to fully leverage LangGraph's capabilities.

## Setup

1. Install the required packages.
2. Set your API keys as environment variables.
3. Set up LangSmith for tracking and debugging.

## Usage

1. Visualize the graph created for the ReAct agent.
2. Interact with the agent by providing user inputs.

## Conclusion

This notebook provides a foundational understanding of how to use a pre-built ReAct agent with LangGraph. For further exploration, consider diving deeper into the [LangGraph documentation](https://langchain-ai.github.io/langgraph/) and experimenting with different agent architectures.