"""Server-only configuration; missing hosted configuration is safe fallback."""
def build_provider(name,tools=None):
    if name=='local':return None
    if name=='gemini':
        from .gemini import GeminiQueryProvider
        return GeminiQueryProvider()
    if name=='microsoft':
        from .microsoft import MicrosoftFoundryProvider
        return MicrosoftFoundryProvider()
    if name=='microsoft-framework':
        from .microsoft import MicrosoftFoundryProvider
        from .agent_framework import FrameworkProvider
        return FrameworkProvider(tools,MicrosoftFoundryProvider())
    raise ValueError('Unsupported configured provider')
