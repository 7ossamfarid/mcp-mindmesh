import { config } from 'dotenv';
import { MindMeshMcpServer, startServer } from './server.js';

// Load environment variables
config();

/**
 * Main entry point for the MindMesh MCP Server
 */
async function main() {
  try {
    console.log("Starting MindMesh MCP Server...");
    
    // Add more detailed logging for initialization steps
    console.log("Environment setup complete. Starting server initialization...");
    
    const server = await startServer()
      .catch(error => {
        console.error("Failed to initialize MindMesh MCP Server:", error);
        // Log more details about the error
        if (error instanceof Error && error.stack) {
          console.error("Error stack:", error.stack);
        }
        
        // Log potential PGlite-specific error information
        if (error.message && error.message.includes("PGlite")) {
          console.error("PGlite error detected. Check database configuration.");
        }
        
        throw error; // Re-throw to be caught by outer catch block
      });
    
    // Handle shutdown signals
    process.on('SIGINT', async () => {
      console.log("Shutting down server...");
      await server.shutdown();
      process.exit(0);
    });
    
    process.on('SIGTERM', async () => {
      console.log("Shutting down server...");
      await server.shutdown();
      process.exit(0);
    });
    
    console.log("MindMesh MCP Server started successfully");
  } catch (error) {
    console.error("Failed to start server:", error);
    // Return a non-zero exit code to indicate failure
    process.exit(1);
  }
}

// Run the server
if (typeof require !== 'undefined' && require.main === module) {
  main();
} else {
  // When used as a module
  main().catch(error => {
    console.error("Error running server:", error);
  });
}

// Export for module usage
export { startServer, MindMeshMcpServer }; 