import os
from functools import cached_property
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.utils import eval_logger
from lm_eval.models.openai_completions import LocalCompletionsAPI, LocalChatCompletion, OpenAICompletionsAPI, OpenAIChatCompletion
import google # pip install google-auth
import vertexai
import openai

from google.auth import default, transport

# Pass the Vertex endpoint and authentication to the OpenAI SDK
PROJECT_ID = os.environ.get('GOOGLE_PROJECT_ID',None)
LOCATION = os.environ.get('LOCATION','us-central1')

# Need to add /chat/completions to the base URL for it to work
@register_model("gemini-openai-chat-completions")
class GeminiOpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url=f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            num_concurrent=2,
            max_retries = 10, # Retry up to 10 times before giving up
            **kwargs,
        )
        self.init_vertex()
        self.current_api_key = None
        self.token_time = time.monotonic()
        print(f"Base URL: {base_url}")


    def get_token(self):

        # Programmatically get an access token
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)
        return credentials.token

    def init_vertex(self):
        if PROJECT_ID is None:
            raise ValueError(
                "PROJECT_ID not found. Please set the `PROJECT_ID` and `LOCATION` environment variable."
            )
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    @property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        # Programmatically get an access token
        # Note: the credential lives for 1 hour by default (https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.
        # Renew token each 20 minutes
        if self.current_api_key is None or time.monotonic() - self.token_time > 1200:
            # renew token
            self.current_api_key = self.get_token()
            self.token_time = time.monotonic()
        if self.current_api_key is None:
            raise ValueError(
                "API key could not be obtained. Please set the `PROJECT_ID` and `LOCATION` environment variable."
            )
        return self.current_api_key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )
