import { Router, type Request, type Response } from 'express';
import { adapterRegistry } from '../../adapters/adapter-registry.js';

const router = Router();

/**
 * GET /api/frameworks
 * List all available frameworks
 */
router.get('/', (req: Request, res: Response) => {
  try {
    const frameworks = adapterRegistry.listAdapters();
    res.json({ frameworks });
  } catch (error) {
    console.error('Error in GET /api/frameworks:', error);
    res.status(500).json({
      error: 'Failed to list frameworks',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/frameworks/:name
 * Get details about a specific framework
 */
router.get('/:name', (req: Request, res: Response) => {
  try {
    const { name } = req.params;

    if (!adapterRegistry.hasAdapter(name)) {
      res.status(404).json({
        error: 'Framework not found',
        framework: name,
        available: adapterRegistry.listAdapters().map((a) => a.name),
      });
      return;
    }

    const adapter = adapterRegistry.getAdapter(name);
    const info = {
      name: adapter.name,
      displayName: adapter.displayName,
      version: adapter.version,
      description: adapter.description,
      capabilities: adapter.getCapabilities?.(),
      configuration: adapter.getConfiguration?.(),
    };

    res.json(info);
  } catch (error) {
    console.error(`Error in GET /api/frameworks/${req.params.name}:`, error);
    res.status(500).json({
      error: 'Failed to get framework details',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

export default router;
