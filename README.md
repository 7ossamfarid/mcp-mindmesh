# MindMesh MCP Server

A Model Context Protocol (MCP) server implementation that creates a quantum-inspired swarm of Claude 3.7 Sonnet instances with field coherence optimization. This server enables enriched reasoning through multiple specialized LLM instances that work together with emergent properties.

## Features

- **Quantum-Inspired Field Computing**: Uses a field-based model to maintain coherence between Claude instances
- **WebContainer Integration**: Full stack sandboxed environment for execution
- **PGLite with Vector Storage**: Efficient vector database with pgvector extension
- **Multiple Claude Specializations**: Instances focus on pattern recognition, information synthesis, and reasoning
- **Coherence Optimization**: Selects the most coherent outputs across instances
- **Extended Thinking Support**: Optional 128k token thinking capability
- **Live Query Updates**: Real-time coherence notifications through PGLite live extension
- **VoyageAI Embeddings**: High-quality embeddings using VoyageAI's state-of-the-art models (voyage-3-large)

## Prerequisites

- Node.js 18.x or higher
- Anthropic API key with access to Claude 3.7 Sonnet
- VoyageAI API key (optional but recommended for better embeddings)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mcp-mindmesh.git
   cd mcp-mindmesh
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Create a `.env` file by copying the template:
   ```
   cp .env.template .env
   ```

4. Edit `.env` and add your Anthropic API key, VoyageAI API key (optional), and adjust other settings as needed.

## Usage

### Starting the Server

Build and start the server:

```
npm run build
npm start
```

For development with auto-reload:

```
npm run dev
```

### Connecting to the Server

You can connect to this MCP server using any MCP client, such as:

1. Claude Desktop Application for Windows (official Anthropic client)
2. Cursor IDE's agent capabilities
3. Cline VSCode extension
4. Any other MCP-compatible client

The server will be available at `http://localhost:3000` by default (or whichever port you specified in the `.env` file).

### Using the Reasoning Tool

The main tool provided by this server is `reason_with_swarm`. This tool takes a prompt and processes it through multiple specialized Claude instances, returning the most coherent result.

Example usage in Claude Desktop:

```
Please use the swarm to analyze the relationship between quantum field theory and consciousness.
```

## Configuration Options

All configuration options can be set in the `.env` file:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | (required) |
| `VOYAGE_API_KEY` | Your VoyageAI API key | (optional) |
| `PORT` | HTTP server port | 3000 |
| `STDIO_TRANSPORT` | Use stdio transport instead of HTTP | false |
| `CLAUDE_INSTANCES` | Number of Claude instances in the swarm | 8 |
| `USE_EXTENDED_THINKING` | Enable 128k extended thinking | true |
| `COHERENCE_THRESHOLD` | Minimum coherence threshold | 0.7 |
| `EMBEDDING_MODEL` | VoyageAI embedding model to use | voyage-3-large |
| `DB_PATH` | Path for the PGLite database | "idb://mindmesh.db" |
| `DEBUG` | Enable debug logging | false |

## Architecture

The server architecture consists of:

1. **MCP Server Layer**: Implements the Model Context Protocol (2025-03-26 specification)
2. **WebContainer Layer**: Provides sandboxed environment for execution
3. **PGLite Vector Database**: Stores state vectors with pgvector extension
4. **Claude Swarm Layer**: Manages multiple specialized Claude instances
5. **Quantum Field Layer**: Handles field coherence and optimization
6. **Embedding Layer**: Generates high-quality embeddings using VoyageAI models

Requests flow through these layers as follows:

```
Client Request → MCP Server → Swarm Processing → Claude API → Coherence Optimization → Response
```

## Advanced Features

### Web Container Integration

The server uses WebContainer technology for a fully sandboxed environment, providing:

- Isolated execution environment
- Full stack capabilities
- File system access
- Network communication

### PGLite with Vector Extension

PGLite provides:

- Client-side PostgreSQL database compiled to WebAssembly
- Vector operations through pgvector extension
- Live query notifications for real-time updates
- Persistent storage across sessions

### Field Coherence Optimization

The coherence optimization system:

1. Processes a query through multiple specialized Claude instances
2. Generates state vectors for each response
3. Calculates coherence metrics between instances
4. Selects the most coherent output
5. Maintains a dynamic field state in the vector database

### VoyageAI Embeddings

The server uses VoyageAI's state-of-the-art embedding models for:

- High-quality state vector generation
- More accurate coherence calculations
- Better field modeling and optimization

When VoyageAI API key is not available, the server falls back to a simpler, deterministic embedding method.

## Development

### Project Structure

- `src/index.ts`: Main entry point
- `src/server.ts`: Core server implementation
- `.env`: Configuration file
- `package.json`: Dependencies and scripts

### Building

```
npm run build
```

This will compile TypeScript to JavaScript in the `dist` directory.

### Testing

```
npm test
```

## License

MIT

## Acknowledgements

This project uses the following technologies:

- [Model Context Protocol](https://modelcontextprotocol.io/) (2025-03-26 spec)
- [Anthropic Claude API](https://www.anthropic.com/)
- [VoyageAI Embeddings](https://voyageai.com/)
- [WebContainer API](https://webcontainers.io/)
- [PGLite](https://pglite.dev/)
- [ElectricSQL vector extension](https://electric-sql.com/) 