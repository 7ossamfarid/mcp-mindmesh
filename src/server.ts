/**
 * MindMesh MCP Server
 * 
 * This implements a Model Context Protocol (MCP) server that creates a quantum-inspired swarm
 * of Claude 3.7 Sonnet instances with field coherence optimization.
 * 
 * Protocol Revision: 2025-03-26
 */

// Use package-level imports, relying on Node module resolution and package.json "exports"
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js"; 
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js"; // Corrected import name
import { z } from "zod";
// import express from 'express'; // Remove express
// import cors from 'cors'; // Remove cors
import http from 'http'; // Keep http
import { URL } from 'url'; // Add URL for parsing
import { WebContainer } from "@webcontainer/api";
import { PGlite, PGliteInterface } from "@electric-sql/pglite"; 
import { live } from '@electric-sql/pglite/live'; // Correct import path for live extension
import { vector } from "@electric-sql/pglite/vector";
import { Anthropic, HUMAN_PROMPT, AI_PROMPT } from "@anthropic-ai/sdk";
import type { Extension } from "@electric-sql/pglite";
import { VoyageAIClient } from "voyageai";

// Core configuration types
interface ClaudeInstance {
  id: string;
  role: 'pattern_recognition' | 'information_synthesis' | 'reasoning';
  model: string;
  apiKey: string;
  stateVector: Float32Array;
  coherence: number;
}

interface CoherenceMetrics {
  overall: number;
  pairwise: Record<string, Record<string, number>>;
  temporal: number[];
}

interface ServerConfig {
  port: number;
  claudeInstances: number;
  dbPath?: string;
  useExtendedThinking: boolean;
  coherenceThreshold: number;
  embeddingModel: string;
  debug: boolean;
}

// Type definition for PGLite notifications
interface PGNotification {
  channel: string;
  payload: string;
}

/**
 * Main MindMesh MCP Server class
 */
export class MindMeshMcpServer {
  private server: McpServer;
  private httpServer: http.Server | null = null;
  private activeTransports: Map<string, SSEServerTransport> = new Map(); // Map to store active SSE transports
  private webcontainer: WebContainer | null = null;
  private db: PGlite | null = null;
  private claudeInstances: Map<string, ClaudeInstance> = new Map();
  private coherenceOptimizer: CoherenceOptimizer;
  private config: ServerConfig;
  private anthropicSdk: Anthropic;
  private _unsubscribers: (() => void)[] = [];

  constructor(config: ServerConfig) {
    this.config = config;
    this.server = new McpServer({
      name: "mindmesh-mcp",
      version: "1.0.0",
      // schema: "2025-03-26" // Removed schema property based on SDK example pattern
    });
    
    this.coherenceOptimizer = new CoherenceOptimizer({
      threshold: config.coherenceThreshold,
      learningRate: 0.05,
      momentum: 0.9
    });
    
    // Transport initialization moved to initialize() for HTTP server setup
    
    this.anthropicSdk = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY || "",
      // Add extended thinking beta header
      defaultHeaders: {
        "anthropic-beta": "output-128k-2025-02-19",
        "anthropic-client": "mindmesh-mcp-server/1.0.0",
        "origin": "https://mindmesh-mcp.webcontainer.io"
      }
    });
  }

  /**
   * Initialize the server and all dependencies
   */
  async initialize(): Promise<void> {
    console.log("Initializing MindMesh MCP Server...");
    
    try {
      // Initialize WebContainer
      if (typeof window !== 'undefined') {
        console.log("Initializing WebContainer...");
        this.webcontainer = await WebContainer.boot();
        console.log("WebContainer initialized");
      } else {
        console.log("WebContainer not available in this environment");
      }
      
      // Initialize PGLite with vector extension
      console.log("Initializing PGLite database...");
      
      // Determine database path/storage mechanism
      let dataDir;
      if (typeof window !== 'undefined') {
        // In browser, use IndexedDB for persistence
        dataDir = `idb://mindmesh-${Date.now()}`;
        console.log(`Using IndexedDB storage: ${dataDir}`);
      } else if (this.config.dbPath) {
        // Use specified path for Node.js environment
        dataDir = this.config.dbPath;
        console.log(`Using filesystem storage: ${dataDir}`);
      } else {
        // Default to in-memory when no path specified
        dataDir = undefined;
        console.log('Using in-memory database (no persistence)');
      }
      
      // Initialize PGlite with proper options
      this.db = await PGlite.create({
        url: dataDir,  // Use url property for database location
        extensions: { 
          vector, 
          live 
        },
        // Use relaxed durability mode for better performance
        relaxedDurability: true
      });
      
      console.log("PGLite database initialized");
      
      // Initialize database schema
      await this.initializeDatabase();
      
      // Create Claude instances
      await this.initializeClaudeInstances();
      
      // Register MCP tools
      this.registerTools();
      
      // Initialize and start HTTP server if not using stdio
      if (process.env.STDIO_TRANSPORT !== "true") {
        const mcpEndpointPath = '/mcp'; // Define the MCP endpoint path

        this.httpServer = http.createServer(async (req, res) => {
          // Use try-catch for overall request handling safety
          try {
            const requestUrl = new URL(req.url || '', `http://${req.headers.host}`);

            if (requestUrl.pathname === mcpEndpointPath) {
              // Handle GET for establishing SSE connection
              if (req.method === 'GET') {
                console.log(`Received GET request on ${mcpEndpointPath} - establishing SSE`);
                // Endpoint client should POST to (relative path)
                const postEndpoint = mcpEndpointPath; 
                const transport = new SSEServerTransport(postEndpoint, res);
                
                this.activeTransports.set(transport.sessionId, transport);
                console.log(`SSE transport created: ${transport.sessionId}`);

                transport.onclose = () => {
                  this.activeTransports.delete(transport.sessionId);
                  console.log(`SSE transport closed: ${transport.sessionId}`);
                };
                transport.onerror = (error) => {
                  console.error(`SSE transport error (${transport.sessionId}):`, error);
                  this.activeTransports.delete(transport.sessionId);
                };

                // Start the transport (sends headers/endpoint event) and connect McpServer
                await transport.start(); 
                await this.server.connect(transport); 
                console.log(`McpServer connected to transport: ${transport.sessionId}`);

              // Handle POST for sending messages
              } else if (req.method === 'POST') {
                const sessionId = requestUrl.searchParams.get('sessionId');
                console.log(`Received POST request on ${mcpEndpointPath} for session: ${sessionId}`);

                if (!sessionId) {
                  res.writeHead(400).end("Missing sessionId query parameter");
                  return;
                }

                const transport = this.activeTransports.get(sessionId);
                if (!transport) {
                  console.warn(`Session not found for POST: ${sessionId}`);
                  res.writeHead(404).end(`Session not found: ${sessionId}`);
                  return;
                }

                // Let the specific transport instance handle the POST message
                await transport.handlePostMessage(req, res);
                console.log(`Handled POST for session: ${sessionId}`);

              // Handle other methods
              } else {
                console.warn(`Method Not Allowed on ${mcpEndpointPath}: ${req.method}`);
                res.writeHead(405).end("Method Not Allowed");
              }
            } else {
              // Handle other paths
              console.log(`Path not found: ${requestUrl.pathname}`);
              res.writeHead(404).end("Not Found");
            }
          } catch (error) {
             console.error("Error handling HTTP request:", error);
             // Ensure response is sent if headers haven't been
             if (!res.writableEnded && !res.headersSent) {
               res.writeHead(500).end("Internal Server Error");
             } else if (!res.writableEnded) {
               // If headers sent (e.g., during SSE setup), just end the response
               res.end(); 
             }
          }
        });

        this.httpServer.listen(this.config.port, () => {
          console.log(`MindMesh MCP Server (HTTP/SSE) initialized and listening on port ${this.config.port}`);
        });

      // Handle stdio transport case
      } else {
        const stdioTransport = new StdioServerTransport();
        await this.server.connect(stdioTransport);
        console.log("MindMesh MCP Server (stdio) initialized and connected.");
      }

      // Set up live query notifications for coherence updates
      await this.setupLiveQueries();
      
    } catch (error) {
      console.error("Failed to initialize MindMesh MCP Server:", error);
      throw error;
    }
  }

  /**
   * Set up live query notifications for coherence updates
   */
  private async setupLiveQueries(): Promise<void> {
    if (!this.db) return;
    
    try {
      // Listen for state vector updates
      await this.db.exec(`
        LISTEN state_vector_updates;
        LISTEN coherence_updates;
      `);
      
      // Use the proper PGlite listen method instead of 'on'
      const unsubStateVectors = await this.db.listen('state_vector_updates', (payload) => {
        try {
          const data = JSON.parse(payload);
          console.log(`State vector update for instance: ${data.instance_id}`);
          // Process state vector updates if needed
        } catch (error) {
          console.error("Error processing state vector update:", error);
        }
      });
      
      const unsubCoherence = await this.db.listen('coherence_updates', (payload) => {
        try {
          const data = JSON.parse(payload);
          console.log(`Coherence update: ${data.overall}`);
          // Could trigger optimizations based on coherence changes
        } catch (error) {
          console.error("Error processing coherence update:", error);
        }
      });
      
      // Store unsubscribe functions for later cleanup if needed
      this._unsubscribers = [unsubStateVectors, unsubCoherence];
      
      console.log("Live queries initialized");
    } catch (error) {
      console.error("Failed to initialize live queries:", error);
    }
  }

  /**
   * Initialize database schema for storing state vectors and coherence
   */
  private async initializeDatabase(): Promise<void> {
    if (!this.db) return;
    
    try {
      // Enable extensions within the database session
      await this.db.exec("CREATE EXTENSION IF NOT EXISTS vector;");
      // await this.db.exec("CREATE EXTENSION IF NOT EXISTS pglite_live;"); // Remove this line - live extension likely enabled via constructor

      // Create state_vectors table with vector support
      await this.db.exec(`
        CREATE TABLE IF NOT EXISTS state_vectors (
          id TEXT PRIMARY KEY,
          instance_id TEXT NOT NULL,
          vector vector(1536) NOT NULL, -- Changed to lowercase 'vector'
          timestamp BIGINT NOT NULL,
          metadata JSONB
        );
      `);
      
      // Create coherence_metrics table
      await this.db.exec(`
        CREATE TABLE IF NOT EXISTS coherence_metrics (
          id TEXT PRIMARY KEY,
          overall REAL NOT NULL,
          pairwise JSONB NOT NULL,
          temporal JSONB NOT NULL,
          timestamp BIGINT NOT NULL
        );
      `);
      
      // Create instances table
      await this.db.exec(`
        CREATE TABLE IF NOT EXISTS instances (
          id TEXT PRIMARY KEY,
          role TEXT NOT NULL,
          model TEXT NOT NULL,
          coherence REAL NOT NULL,
          last_active BIGINT NOT NULL
        );
      `);
      
      // Create indices
      await this.db.exec(`
        CREATE INDEX IF NOT EXISTS idx_state_vectors_instance_id ON state_vectors(instance_id);
        CREATE INDEX IF NOT EXISTS idx_coherence_metrics_timestamp ON coherence_metrics(timestamp);
      `);
      
      // Create trigger for coherence updates
      await this.db.exec(`
        CREATE OR REPLACE FUNCTION notify_coherence_update()
        RETURNS TRIGGER AS $$
        BEGIN
          PERFORM pg_notify('coherence_updates', row_to_json(NEW)::text);
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS coherence_update_trigger ON coherence_metrics;
        CREATE TRIGGER coherence_update_trigger
        AFTER INSERT OR UPDATE ON coherence_metrics
        FOR EACH ROW
        EXECUTE FUNCTION notify_coherence_update();
      `);
      
      console.log("Database schema initialized");
    } catch (error) {
      console.error("Failed to initialize database schema:", error);
      throw error;
    }
  }

  /**
   * Initialize Claude instances with different specializations
   */
  private async initializeClaudeInstances(): Promise<void> {
    const roles: Array<'pattern_recognition' | 'information_synthesis' | 'reasoning'> = [
      'pattern_recognition',
      'information_synthesis',
      'reasoning'
    ];
    
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error("ANTHROPIC_API_KEY environment variable not set");
    }
    
    // Create instances
    for (let i = 0; i < this.config.claudeInstances; i++) {
      const role = roles[i % roles.length];
      const id = `claude_${role}_${i}`;
      
      const instance: ClaudeInstance = {
        id,
        role,
        model: "claude-3-7-sonnet-20250219",
        apiKey,
        stateVector: new Float32Array(1536),
        coherence: 1.0
      };
      
      this.claudeInstances.set(id, instance);
      
      // Store in database
      if (this.db) {
        // Use proper query format
        await this.db.query(
          `INSERT INTO instances (id, role, model, coherence, last_active)
          VALUES ($1, $2, $3, $4, $5)
          ON CONFLICT (id) DO UPDATE SET
            role = $2,
            model = $3,
            coherence = $4,
            last_active = $5`,
          [id, role, instance.model, instance.coherence, Date.now()]
        );
      }
    }
    
    console.log(`Initialized ${this.claudeInstances.size} Claude instances`);
  }

  /**
   * Register MCP tools
   */
  private registerTools(): void {
    // Register the main reasoning tool following MCP SDK format
    this.server.tool(
      "reason_with_swarm",
      // Use schema as second parameter in the proper format
      {
        prompt: z.string(),
        temperature: z.number().min(0).max(1).optional(),
        topK: z.number().int().min(1).max(60).optional(),
        topP: z.number().min(0.85).max(1).optional(),
        useExtendedThinking: z.boolean().optional()
      },
      // Add explicit types
      async (args: any, context: any) => {
        try {
          if (this.config.debug) {
            console.log("Received request:", args);
          }
          
          // Process the request with the swarm
          const result = await this.processWithSwarm(
            args.prompt,
            args.temperature || 0.7,
            args.topK || 40,
            args.topP || 0.95,
            args.useExtendedThinking ?? this.config.useExtendedThinking
          );
          
          // Format response according to MCP spec
          return {
            content: [
              {
                type: "text",
                text: result.response
              }
            ],
            metadata: {
              coherence: result.coherence,
              instances: result.instances
            }
          };
        } catch (error) {
          console.error("Error processing with swarm:", error);
          return {
            content: [
              {
                type: "text",
                text: `Error: ${error instanceof Error ? error.message : String(error)}`
              }
            ],
            isError: true
          };
        }
      }
    );
    
    // Add list_instances tool
    this.server.tool(
      "list_instances",
      {}, // Empty schema object
      async () => {
        const instances = Array.from(this.claudeInstances.entries()).map(([id, instance]) => ({
          id,
          role: instance.role,
          model: instance.model,
          coherence: instance.coherence
        }));
        return {
          content: [{ type: "text", text: JSON.stringify(instances, null, 2) }]
        };
      }
    );

    // Add measure_coherence tool
    this.server.tool(
      "measure_coherence",
      {}, // Empty schema object
      async () => {
        const metrics = await this.coherenceOptimizer.calculateCoherence();
        return {
          content: [{ type: "text", text: JSON.stringify(metrics, null, 2) }]
        };
      }
    );

    // Keep health check tool
    this.server.tool(
      "health_check",
      {}, // Empty schema object
      async () => {
        const coherence = await this.coherenceOptimizer.calculateCoherence();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                status: "ok",
                instances: this.claudeInstances.size,
                coherence: coherence.overall,
                timestamp: Date.now()
              }, null, 2)
            }
          ]
        };
      }
    );
  }

  /**
   * Process a prompt with the entire swarm
   */
  private async processWithSwarm(
    prompt: string,
    temperature: number,
    topK: number,
    topP: number,
    useExtendedThinking: boolean
  ): Promise<{
    response: string;
    coherence: number;
    instances: Array<{id: string; role: string}>;
  }> {
    // Prepare concurrent requests for all instances
    const instanceProcessing = Array.from(this.claudeInstances.entries()).map(
      async ([id, instance]) => {
        try {
          const specializedPrompt = this.createSpecializedPrompt(instance.role, prompt);
          
          const result = await this.processWithInstance(
            instance,
            specializedPrompt,
            temperature,
            topK,
            topP,
            useExtendedThinking
          );
          
          // Store state vector in database
          if (this.db) {
            const vectorId = `${instance.id}_${this.simpleHash(prompt)}_${Date.now()}`;
            // Use proper query format
            await this.db.query(
              `INSERT INTO state_vectors (id, instance_id, vector, timestamp, metadata)
              VALUES ($1, $2, $3, $4, $5)`,
              [
                vectorId,
                instance.id,
                result.stateVector,
                Date.now(),
                JSON.stringify({ role: instance.role, prompt })
              ]
            );
          }
          
          this.coherenceOptimizer.registerStateVector(instance.id, result.stateVector, {
            role: instance.role,
            prompt
          });
          
          return {
            instanceId: instance.id,
            role: instance.role,
            response: result.response,
            stateVector: result.stateVector
          };
        } catch (error) {
          console.error(`Error processing with instance ${id}:`, error);
          // Return partial result on error
          return {
            instanceId: instance.id,
            role: instance.role,
            response: `Error processing with ${instance.role} instance: ${error}`,
            stateVector: new Float32Array(1536)
          };
        }
      }
    );
    
    // Wait for all instances to complete
    const results = await Promise.all(instanceProcessing);
    
    // Get responses and state vectors
    const responses = results.map(result => result.response);
    const stateVectors = results.map(result => result.stateVector);
    
    // Optimize outputs
    const optimized = await this.coherenceOptimizer.optimizeOutputs(responses, stateVectors);
    
    // Store coherence metrics
    if (this.db) {
      const coherence = await this.coherenceOptimizer.calculateCoherence();
      // Use proper query format
      await this.db.query(
        `INSERT INTO coherence_metrics (id, overall, pairwise, temporal, timestamp)
        VALUES ($1, $2, $3, $4, $5)`,
        [
          `coherence_${Date.now()}`,
          coherence.overall,
          JSON.stringify(coherence.pairwise),
          JSON.stringify(coherence.temporal),
          Date.now()
        ]
      );
    }
    
    return {
      response: optimized.optimizedOutput,
      coherence: optimized.coherence,
      instances: results.map(result => ({
        id: result.instanceId,
        role: result.role
      }))
    };
  }

  /**
   * Process a prompt with a specific Claude instance
   */
  private async processWithInstance(
    instance: ClaudeInstance,
    prompt: string,
    temperature: number,
    topK: number | undefined,
    topP: number | undefined,
    useExtendedThinking: boolean
  ): Promise<{
    response: string;
    stateVector: Float32Array;
  }> {
    try {
      // Prepare the message with system prompt based on instance role
      const systemPrompt = this.getSystemPrompt(instance.role);
      
      const response = await this.anthropicSdk.messages.create({
        model: instance.model,
        system: systemPrompt,
        messages: [
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: temperature,
        top_k: topK,
        top_p: topP,
        max_tokens: useExtendedThinking ? 128000 : 4096
      });
      
      const responseText = response.content.map(c => {
        if (c.type === 'text') {
          return c.text;
        }
        return '';
      }).join('');
      
      // Generate state vector for response
      const stateVector = await this.generateStateVector(responseText);
      
      return {
        response: responseText,
        stateVector
      };
    } catch (error) {
      console.error(`Error calling Claude API for instance ${instance.id}:`, error);
      throw error;
    }
  }

  /**
   * Get the system prompt for a specific role
   */
  private getSystemPrompt(role: string): string {
    switch (role) {
      case 'pattern_recognition':
        return `You are a specialized Claude instance focused on pattern recognition. 
          Your strength is identifying patterns, connections, and structure in information.
          Focus on extracting insights through pattern analysis.`;
          
      case 'information_synthesis':
        return `You are a specialized Claude instance focused on information synthesis.
          Your strength is combining and integrating diverse pieces of information into a coherent whole.
          Focus on creating integrated understanding from multiple sources and perspectives.`;
          
      case 'reasoning':
        return `You are a specialized Claude instance focused on logical reasoning.
          Your strength is applying step-by-step logical analysis, causal reasoning, and problem-solving.
          Focus on developing conclusions through careful, structured reasoning.`;
          
      default:
        return `You are a specialized Claude instance in a quantum-inspired swarm.
          Work to provide clear, accurate, and helpful information with your unique perspective.`;
    }
  }

  /**
   * Create a specialized prompt based on instance role
   */
  private createSpecializedPrompt(role: string, basePrompt: string): string {
    switch (role) {
      case 'pattern_recognition':
        return `${basePrompt}
          
          As a pattern recognition specialist, identify any important patterns, correlations, structures, or recurring themes that are relevant. What connections might others miss?`;
          
      case 'information_synthesis':
        return `${basePrompt}
          
          As an information synthesis specialist, integrate all relevant knowledge and perspectives into a coherent whole. What is the most complete and integrated understanding?`;
          
      case 'reasoning':
        return `${basePrompt}
          
          As a reasoning specialist, apply careful logical analysis to this situation. What conclusions follow from the most rigorous reasoning process?`;
          
      default:
        return basePrompt;
    }
  }

  /**
   * Generate a state vector for a text using Voyage AI embeddings
   */
  private async generateStateVector(text: string): Promise<Float32Array> {
    try {
      const voyageApiKey = process.env.VOYAGE_API_KEY;
      
      if (!voyageApiKey) {
        console.warn("VOYAGE_API_KEY not set, using fallback embedding method");
        return this.generateFallbackStateVector(text);
      }
      
      const voyageClient = new VoyageAIClient({ apiKey: voyageApiKey });
      
      // Call Voyage API to get embeddings
      // Use any type to handle different response formats
      const response: any = await voyageClient.embed({
        input: text,
        model: this.config.embeddingModel || "voyage-3",
        inputType: "document",
        truncation: true
      });

      // Check for embedding in data array - format should be { data: [{ embedding: [...] }] }
      if (response.data && 
          Array.isArray(response.data) && 
          response.data.length > 0 && 
          response.data[0].embedding) {
        return new Float32Array(response.data[0].embedding);
      } 
      
      // Alternative response format might be { embeddings: [...] }
      else if (response.embeddings && 
               Array.isArray(response.embeddings) && 
               response.embeddings.length > 0) {
        return new Float32Array(response.embeddings[0]);
      }
      
      // Log the actual response structure and fall back
      console.warn("Unexpected Voyage AI response structure:", JSON.stringify(response, null, 2));
      return this.generateFallbackStateVector(text);
    } catch (error) {
      console.error("Error generating embeddings with Voyage API:", error);
      // Fall back to simpler approach if API call fails
      return this.generateFallbackStateVector(text);
    }
  }

  /**
   * Fallback method for generating state vectors when the embedding API is unavailable
   */
  private generateFallbackStateVector(text: string): Float32Array {
    // Create a vector of the right size (1536 dimensions)
    const vector = new Float32Array(1536);
    
    // Fill with pseudo-random values based on the text
    const seed = this.simpleHash(text);
    for (let i = 0; i < 1536; i++) {
      // Pseudorandom but deterministic value between -1 and 1
      const value = Math.sin(seed * i) * Math.cos(i);
      vector[i] = value;
    }
    
    // Normalize the vector to unit length
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) { // Avoid division by zero
        for (let i = 0; i < vector.length; i++) {
          vector[i] /= magnitude;
        }
    }
    
    return vector;
  }

  /**
   * Simple hash function for text
   */
  private simpleHash(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash;
  }

  /**
   * Shut down the server and clean up resources
   */
  async shutdown(): Promise<void> {
    console.log("Shutting down MindMesh MCP Server...");
    
    try {
      // Close HTTP server if it exists
      if (this.httpServer) {
        await new Promise<void>((resolve) => {
          this.httpServer!.close(() => resolve());
        });
        console.log("HTTP server closed");
      }
      
      // For each active transport, safely close it if possible
      for (const [sessionId, transport] of this.activeTransports.entries()) {
        try {
          console.log(`Closing SSE transport: ${sessionId}`);
          transport.close();
        } catch (error) {
          console.error(`Error closing transport ${sessionId}:`, error);
        }
      }
      this.activeTransports.clear();
      
      // Disconnect MCP server
      try {
        // Use connect with null transport to disconnect or other equivalent method
        // based on the SDK implementation
        console.log("Disconnecting MCP server...");
        // await this.server.disconnect(); // Previous incorrect method
        
        // Alternative approaches depending on SDK version:
        // 1. Set server to null or recreate it
        this.server = new McpServer({
          name: "mindmesh-mcp",
          version: "1.0.0"
        });
      } catch (error) {
        console.error("Error disconnecting MCP server:", error);
      }
      
      // Close database connection if it exists
      if (this.db) {
        try {
          console.log("Closing database connection...");
          await this.db.close();
          this.db = null;
        } catch (error) {
          console.error("Error closing database connection:", error);
        }
      }
      
      // Release WebContainer resources if it exists
      if (this.webcontainer) {
        try {
          console.log("Releasing WebContainer resources...");
          // Any cleanup needed for webcontainer
          this.webcontainer = null;
        } catch (error) {
          console.error("Error releasing WebContainer resources:", error);
        }
        console.log("WebContainer resources released");
      }
      
      // Clean up live query listeners
      for (const unsub of this._unsubscribers) {
        unsub();
      }
      this._unsubscribers = [];
      
      console.log("MindMesh MCP Server shut down successfully");
    } catch (error) {
      console.error("Error shutting down MindMesh MCP Server:", error);
      throw error;
    }
  }
}

/**
 * Coherence Optimizer class for managing state vectors and optimizing outputs
 */
class CoherenceOptimizer {
  private stateVectors: Map<string, {
    vector: Float32Array;
    metadata: any;
  }> = new Map();
  
  private gradientHistory: Map<string, Float32Array> = new Map();
  private threshold: number;
  private learningRate: number;
  private momentum: number;
  
  constructor(options: {
    threshold: number;
    learningRate: number;
    momentum: number;
  }) {
    this.threshold = options.threshold;
    this.learningRate = options.learningRate;
    this.momentum = options.momentum;
  }
  
  /**
   * Register a state vector for an instance
   */
  registerStateVector(id: string, vector: Float32Array, metadata: any): void {
    this.stateVectors.set(id, {
      vector,
      metadata
    });
  }
  
  /**
   * Calculate coherence metrics for all registered state vectors
   */
  async calculateCoherence(): Promise<CoherenceMetrics> {
    // Calculate pairwise coherence
    const pairwise: Record<string, Record<string, number>> = {};
    const ids = Array.from(this.stateVectors.keys());
    
    let totalCoherence = 0;
    let numPairs = 0;
    
    // Calculate pairwise similarities
    for (let i = 0; i < ids.length; i++) {
      const id1 = ids[i];
      pairwise[id1] = {};
      
      for (let j = i + 1; j < ids.length; j++) {
        const id2 = ids[j];
        
        const vec1 = this.stateVectors.get(id1)?.vector;
        const vec2 = this.stateVectors.get(id2)?.vector;

        if (vec1 && vec2) { // Ensure vectors exist
            const similarity = this.calculateCosineSimilarity(vec1, vec2);
            
            pairwise[id1][id2] = similarity;
            totalCoherence += similarity;
            numPairs++;
            
            // Set symmetric relation
            if (!pairwise[id2]) {
              pairwise[id2] = {};
            }
            pairwise[id2][id1] = similarity;
        }
      }
    }
    
    // Calculate overall coherence
    const overall = numPairs > 0 ? totalCoherence / numPairs : 0;
    
    // Temporal coherence is empty for now (would track changes over time)
    const temporal: number[] = [];
    
    return {
      overall,
      pairwise,
      temporal
    };
  }
  
  /**
   * Calculate cosine similarity between two vectors
   */
  private calculateCosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error("Vectors must have the same length");
    }
    
    // Calculate dot product
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magnitudeA += a[i] * a[i];
      magnitudeB += b[i] * b[i];
    }
    
    // Calculate magnitudes
    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);
    
    // Return cosine similarity
    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }
    
    return dotProduct / (magnitudeA * magnitudeB);
  }
  
  /**
   * Optimize outputs based on coherence
   */
  async optimizeOutputs(
    outputs: string[],
    stateVectors: Float32Array[]
  ): Promise<{
    optimizedOutput: string;
    coherence: number;
  }> {
    if (outputs.length !== stateVectors.length) {
      throw new Error("Number of outputs must match number of state vectors");
    }
    
    if (outputs.length === 0) {
      return {
        optimizedOutput: "",
        coherence: 0
      };
    }
    
    if (outputs.length === 1) {
      return {
        optimizedOutput: outputs[0],
        coherence: 1.0
      };
    }
    
    // Calculate pairwise similarities between state vectors
    const similarities: number[][] = [];
    for (let i = 0; i < stateVectors.length; i++) {
      similarities[i] = [];
      for (let j = 0; j < stateVectors.length; j++) {
        if (i === j) {
          similarities[i][j] = 1.0; // Self-similarity is 1
        } else {
          similarities[i][j] = this.calculateCosineSimilarity(stateVectors[i], stateVectors[j]);
        }
      }
    }
    
    // Calculate coherence score for each output
    const scores = similarities.map(row => row.reduce((sum, val) => sum + val, 0) / row.length);
    
    // Find the output with highest coherence
    let maxScore = -Infinity;
    let maxIndex = 0;
    
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIndex = i;
      }
    }
    
    // Return the most coherent output
    return {
      optimizedOutput: outputs[maxIndex],
      coherence: maxScore
    };
  }
  
  /**
   * Get metrics for the optimizer
   */
  getMetrics(): any {
    return {
      numVectors: this.stateVectors.size,
      threshold: this.threshold,
      learningRate: this.learningRate,
      momentum: this.momentum
    };
  }
}

/**
 * Start the server with the given configuration
 */
export async function startServer(config: Partial<ServerConfig> = {}): Promise<MindMeshMcpServer> {
  // Set defaults for configuration
  const fullConfig: ServerConfig = {
    port: config.port || Number(process.env.PORT) || 3000,
    claudeInstances: config.claudeInstances || Number(process.env.CLAUDE_INSTANCES) || 3,
    dbPath: config.dbPath || process.env.DB_PATH || undefined,
    useExtendedThinking: config.useExtendedThinking ?? (process.env.USE_EXTENDED_THINKING === 'true'),
    coherenceThreshold: config.coherenceThreshold || Number(process.env.COHERENCE_THRESHOLD) || 0.85,
    embeddingModel: config.embeddingModel || process.env.EMBEDDING_MODEL || 'voyage-3',
    debug: config.debug ?? (process.env.DEBUG === 'true')
  };
  
  console.log("Creating MindMesh MCP Server with config:", JSON.stringify(fullConfig, null, 2));
  
  try {
    // Create server instance
    const server = new MindMeshMcpServer(fullConfig);
    
    // Initialize server
    console.log("Initializing server...");
    await server.initialize();
    
    return server;
  } catch (error) {
    console.error("Failed to start server:", error);
    
    // Additional error diagnostics
    if (error instanceof Error) {
      if (error.message.includes('PGlite') || error.message.includes('database')) {
        console.error("Database initialization error. Check PGlite configuration.");
        if (typeof window !== 'undefined') {
          console.log("Browser environment detected. Ensure IndexedDB is available and not blocked.");
        } else {
          console.log("Node.js environment detected. Check filesystem permissions for database path.");
        }
      }
      
      if (error.message.includes('WebContainer')) {
        console.error("WebContainer initialization error. Ensure browser environment supports WebContainers.");
      }
      
      if (error.message.includes('ANTHROPIC_API_KEY')) {
        console.error("API key configuration error. Ensure ANTHROPIC_API_KEY is set in environment variables.");
      }
    }
    
    throw error;
  }
}
