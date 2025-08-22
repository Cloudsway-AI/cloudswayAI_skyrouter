from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMMode
import requests
import json
import yaml
import os
from urllib.parse import urljoin
from dify_plugin.errors.model import CredentialsValidateFailedError

# thinking models compatibility for max_completion_tokens (all starting with "o" or "gpt-5")
THINKING_SERIES_COMPATIBILITY = ("o", "gpt-5")

class _Common:
    """
        Common methods for updating credentials
    """
    
    def _get_model_yaml_path(self, model: str) -> str:
        """
        Get the path to the model's YAML file based on model name
        """
        # Extract the model identifier from the full model name
        # e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini"
        model_id = model.split('/')[-1] if '/' in model else model
        
        # Get the directory where this common.py file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, f"{model_id}.yaml")
        
        return yaml_path
    
    def _load_model_from_yaml(self, model: str) -> dict:
        """
        Load model configuration directly from YAML file
        """
        yaml_path = self._get_model_yaml_path(model)
        
        if not os.path.exists(yaml_path):
            # Try to find a matching YAML file by partial name
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_id = model.split('/')[-1] if '/' in model else model
            for filename in os.listdir(current_dir):
                if filename.endswith('.yaml') and model_id in filename:
                    yaml_path = os.path.join(current_dir, filename)
                    break
            else:
                raise FileNotFoundError(f"Model YAML file not found for {model}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                model_config = yaml.safe_load(file)
            return model_config
        except Exception as e:
            raise Exception(f"Failed to load model YAML file {yaml_path}: {e}")
    
    def _get_model_features_from_yaml(self, model: str) -> list:
        """
        Get model features directly from YAML file
        """
        model_config = self._load_model_from_yaml(model)
        return model_config.get('features', [])
    
    def _get_model_properties_from_yaml(self, model: str) -> dict:
        """
        Get model properties directly from YAML file
        """
        model_config = self._load_model_from_yaml(model)
        return model_config.get('model_properties', {})
    
    def _get_parameter_rules_from_yaml(self, model: str) -> list:
        """
        Get parameter rules directly from YAML file
        """
        model_config = self._load_model_from_yaml(model)
        return model_config.get('parameter_rules', [])
    
    def _get_pricing_from_yaml(self, model: str) -> dict:
        """
        Get pricing directly from YAML file
        """
        model_config = self._load_model_from_yaml(model)
        return model_config.get('pricing', {})

    def _update_credential(self, model: str, credentials: dict):
        credentials["mode"] = LLMMode.CHAT.value
        schema = self.get_model_schema(model, credentials)
        if schema and {ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL}.intersection(
            schema.features or []
        ):
            credentials["function_calling_type"] = "tool_call"
            
            
    def _validate_credentials_common(self, model: str, credentials: dict):
        """
        Common credential validation logic
        """

        try:
            headers = {"Content-Type": "application/json"}

            api_key = credentials.get("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            endpoint_url = credentials["endpoint_url"]
            if not endpoint_url.endswith("/"):
                endpoint_url += "/"

            data = {}
            
            # prepare the payload for a simple ping to the model
            if model.startswith(THINKING_SERIES_COMPATIBILITY):
                data["max_completion_tokens"] = 5
            else:
                data["max_tokens"] = 5


            completion_type = LLMMode.value_of(credentials["mode"])

            if completion_type is LLMMode.CHAT:
                data["messages"] = [
                    {"role": "user", "content": "ping"},
                ]
                endpoint_url = urljoin(endpoint_url, "chat/completions")
            elif completion_type is LLMMode.COMPLETION:
                data["prompt"] = "ping"
                endpoint_url = urljoin(endpoint_url, "completions")
            else:
                raise ValueError("Unsupported completion type for model configuration.")

            # ADD stream validate_credentials
            stream_mode_auth = credentials.get("stream_mode_auth", "not_use")
            if stream_mode_auth == "use":
                data["stream"] = True
                data["max_tokens"] = 10
                response = requests.post(endpoint_url, headers=headers, json=data, timeout=(10, 300), stream=True)
                if response.status_code != 200:
                    raise CredentialsValidateFailedError(
                        f"Credentials validation failed with status code {response.status_code} "
                        f"and response body {response.text}"
                    )
                return

            # send a post request to validate the credentials
            response = requests.post(endpoint_url, headers=headers, json=data, timeout=(10, 300))

            print(response)
                
            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed with status code {response.status_code} "
                    f"and response body {response.text}"
                )

            try:
                json_result = response.json()
            except json.JSONDecodeError:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed: JSON decode error, response body {response.text}"
                ) from None

            if completion_type is LLMMode.CHAT and json_result.get("object", "") == "":
                json_result["object"] = "chat.completion"
            elif completion_type is LLMMode.COMPLETION and json_result.get("object", "") == "":
                json_result["object"] = "text_completion"

            if completion_type is LLMMode.CHAT and (
                "object" not in json_result or json_result["object"] != "chat.completion"
            ):
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed: invalid response object, "
                    f"must be 'chat.completion', response body {response.text}"
                )
            elif completion_type is LLMMode.COMPLETION and (
                "object" not in json_result or json_result["object"] != "text_completion"
            ):
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed: invalid response object, "
                    f"must be 'text_completion', response body {response.text}"
                )
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            # Fix the undefined 'response' variable issue
            error_msg = f"An error occurred during credentials validation: {ex!s}"
            if 'response' in locals():
                error_msg += f", response body {response.text}"
            raise CredentialsValidateFailedError(error_msg) from ex