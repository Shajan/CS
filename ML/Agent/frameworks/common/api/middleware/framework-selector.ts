import type { Request, Response, NextFunction } from 'express';
import { adapterRegistry } from '../../adapters/adapter-registry.js';

/**
 * Middleware to extract and validate framework from request
 * Attaches the adapter to the request object for use in routes
 */
export function frameworkSelector(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  try {
    // Get framework from request body or default to 'claude-agent'
    const frameworkName = req.body.framework || 'claude-agent';

    // Check if framework exists
    if (!adapterRegistry.hasAdapter(frameworkName)) {
      res.status(404).json({
        error: 'Framework not found',
        framework: frameworkName,
        available: adapterRegistry.listAdapters().map((a) => a.name),
      });
      return;
    }

    // Attach adapter to request
    const adapter = adapterRegistry.getAdapter(frameworkName);
    (req as any).adapter = adapter;
    (req as any).frameworkName = frameworkName;

    next();
  } catch (error) {
    res.status(500).json({
      error: 'Failed to select framework',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
