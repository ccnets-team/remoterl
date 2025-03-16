from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

CURRENT_REMOTE_RL_VERSION = "latest"  # Current Image Tag in ACCESIBLE_REGIONS

@dataclass
class TrainerConfig:
    DEFAULT_OUTPUT_PATH = "s3://<your-output-path>/"
    instance_type: str = "ml.g5.4xlarge"
    instance_count: int = 1
    max_run: int = 3600
    output_path: Optional[str] = DEFAULT_OUTPUT_PATH

@dataclass
class InferenceConfig:
    DEFAULT_MODEL_DATA = "s3://your-bucket/model.tar.gz"
    endpoint_name: Optional[str] = "remoterl-inference-endpoint"
    instance_type: str = "ml.t2.medium"
    instance_count: int = 1
    max_run: int = 3600
    model_data: Optional[str] = DEFAULT_MODEL_DATA
    
@dataclass
class SageMakerConfig:
    role_arn: Optional[str] = "arn:aws:iam::<your-aws-account-id>:role/SageMakerExecutionRole"
    region: Optional[str] = "<your-aws-region>"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def __post_init__(self):
        # Convert nested dictionaries to their respective dataclass instances if needed.
        if isinstance(self.trainer, dict):
            self.trainer = TrainerConfig(**self.trainer)
        if isinstance(self.inference, dict):
            self.inference = InferenceConfig(**self.inference)

    def get_image_uri(self, service_type: str) -> str:
        if service_type not in ("trainer", "inference"):
            raise ValueError("service_type must be either 'trainer' or 'inference'")
        
        # Construct the image URI dynamically based on region and service type.
        return f"533267316703.dkr.ecr.{self.region}.amazonaws.com/remoterl-{service_type}:{CURRENT_REMOTE_RL_VERSION}"

    def to_dict(self) -> Dict:
        """Returns a nested dictionary of the full SageMaker configuration."""
        return asdict(self)
    
    def set_config(self, **kwargs):
        """
        Update the SageMakerConfig instance using provided keyword arguments.
        For nested fields like 'trainer' and 'inference', update only the specified sub-attributes.
        """
        for k, v in kwargs.items():
            if k == "trainer" and isinstance(v, dict):
                for sub_key, sub_value in v.items():
                    if hasattr(self.trainer, sub_key):
                        setattr(self.trainer, sub_key, sub_value)
                    else:
                        print(f"Warning: TrainerConfig has no attribute '{sub_key}'")
            elif k == "inference" and isinstance(v, dict):
                for sub_key, sub_value in v.items():
                    if hasattr(self.inference, sub_key):
                        setattr(self.inference, sub_key, sub_value)
                    else:
                        print(f"Warning: InferenceConfig has no attribute '{sub_key}'")
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in SageMakerConfig")