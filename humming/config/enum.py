import enum


class InputQuantizationMode(str, enum.Enum):
    Disabled = "none"
    StaticTensor = "static_tensor"
    DynamicToken = "dynamic_token"
    DynamicGroup = "dynamic_group"
    StaticTensorDynamicGroup = "static_tensor_dynamic_group"
    DynamicGroupToken = "dynamic_group_token"

    @property
    def should_quantize(self) -> bool:
        return self != InputQuantizationMode.Disabled

    @property
    def has_static_tensor_scale(self) -> bool:
        return self in (InputQuantizationMode.StaticTensor, InputQuantizationMode.StaticTensorDynamicGroup)

    @property
    def has_dynamic_token_scale(self) -> bool:
        return self in (InputQuantizationMode.DynamicToken, InputQuantizationMode.DynamicGroupToken)

    @property
    def has_dynamic_group_scale(self) -> bool:
        return self in (
            InputQuantizationMode.DynamicGroup,
            InputQuantizationMode.StaticTensorDynamicGroup,
            InputQuantizationMode.DynamicGroupToken,
        )

    @property
    def has_dynamic_scale(self) -> bool:
        return self.has_dynamic_token_scale or self.has_dynamic_group_scale

    @property
    def dynamic_scale_mode(self) -> str | None:
        if self.has_dynamic_token_scale:
            return "group_token" if self.has_dynamic_group_scale else "token"
        if self.has_dynamic_group_scale:
            return "group"
        return None

    @property
    def uses_token_scale(self) -> bool:
        return self.has_dynamic_token_scale

    @property
    def uses_tensor_scale(self) -> bool:
        return self.has_static_tensor_scale

    @property
    def uses_group_scale(self) -> bool:
        return self.has_dynamic_group_scale

    @property
    def has_secondary_scale(self) -> bool:
        return self.uses_group_scale and (self.uses_tensor_scale or self.uses_token_scale)


class MmaType(enum.Enum):
    MMA = "mma"
    WGMMA = "wgmma"
    MXMMA = "mxmma"


class WeightScaleType(enum.Enum):
    GROUP = "group"
    BLOCK = "block"
    CHANNEL = "channel"
    TENSOR = "tensor"


class WeightScale2Type(enum.Enum):
    NONE = "none"
    CHANNEL = "channel"
    TENSOR = "tensor"


class GemmType(enum.Enum):
    DENSE = "dense"
    INDEXED = "indexed"
    GROUPED_CONTIGUOUS = "grouped_contiguous"
    GROUPED_MASKED = "grouped_masked"
