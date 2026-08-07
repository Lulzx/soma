//! Logical discovery nodes. These are not SOMA ABI entities.

use crate::compiler::body::EvaluatorProgram;

use super::key::{hash_fields, ExperimentKey, ModuleDigest, NodeDigest, ObjectDigest};

pub type HypothesisId = u64;
pub type RequestId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FusionClass {
    Pointwise,
    LocalGather,
    Unfusible,
}

impl FusionClass {
    /// Derive fusion safety from the evaluator rather than trusting a caller.
    pub fn for_program(program: &EvaluatorProgram) -> Self {
        if program.gathers() || program.binds_aux() {
            FusionClass::LocalGather
        } else {
            FusionClass::Pointwise
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationSpec {
    pub operation: String,
    pub module: ModuleDigest,
    pub evaluator_id: u32,
    pub inputs: Vec<u8>,
    pub aux_inputs: Vec<u8>,
    pub aux_element_count: u32,
    pub aux_element_stride: u32,
    pub parameters: Vec<u8>,
    pub contract: Vec<u8>,
    pub seed: u64,
    pub element_count: u32,
    pub element_stride: u32,
    pub fusion: FusionClass,
}

impl EvaluationSpec {
    pub fn new(
        operation: impl Into<String>,
        module: ModuleDigest,
        evaluator_id: u32,
        inputs: Vec<u8>,
        element_count: u32,
        element_stride: u32,
        fusion: FusionClass,
    ) -> Self {
        Self {
            operation: operation.into(),
            module,
            evaluator_id,
            inputs,
            aux_inputs: Vec::new(),
            aux_element_count: 0,
            aux_element_stride: 0,
            parameters: Vec::new(),
            contract: Vec::new(),
            seed: 0,
            element_count,
            element_stride,
            fusion,
        }
    }

    pub fn input_digest(&self) -> ObjectDigest {
        ObjectDigest::of(&self.inputs)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.operation.is_empty() {
            return Err("empty discovery operation");
        }
        if self.element_stride == 0 {
            return Err("zero discovery element stride");
        }
        let required = (self.element_count as usize)
            .checked_mul(self.element_stride as usize)
            .ok_or("discovery input size overflow")?;
        if self.inputs.len() != required {
            return Err("discovery input shape mismatch");
        }
        let aux_absent = self.aux_element_count == 0
            && self.aux_element_stride == 0
            && self.aux_inputs.is_empty();
        if !aux_absent {
            if self.aux_element_stride == 0 {
                return Err("zero discovery aux stride");
            }
            let required = (self.aux_element_count as usize)
                .checked_mul(self.aux_element_stride as usize)
                .ok_or("discovery aux size overflow")?;
            if self.aux_inputs.len() != required {
                return Err("discovery aux shape mismatch");
            }
            if self.fusion == FusionClass::Pointwise {
                return Err("pointwise discovery node binds an aux array");
            }
        }
        Ok(())
    }

    pub fn semantic_key(&self, kind: &[u8]) -> ExperimentKey {
        let input = self.input_digest().0;
        let aux = ObjectDigest::of(&self.aux_inputs).0;
        let evaluator = self.evaluator_id.to_le_bytes();
        let seed = self.seed.to_le_bytes();
        let count = self.element_count.to_le_bytes();
        let stride = self.element_stride.to_le_bytes();
        let aux_count = self.aux_element_count.to_le_bytes();
        let aux_stride = self.aux_element_stride.to_le_bytes();
        ExperimentKey(hash_fields(
            b"soma.discovery.experiment.v1",
            &[
                kind,
                self.operation.as_bytes(),
                &self.module.0,
                &evaluator,
                &input,
                &aux,
                &aux_count,
                &aux_stride,
                &self.parameters,
                &self.contract,
                &seed,
                &count,
                &stride,
            ],
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryNode {
    Derivation(EvaluationSpec),
    Observation {
        sample: u64,
        evaluation: EvaluationSpec,
    },
    Aggregate(EvaluationSpec),
    Decision(EvaluationSpec),
}

impl DiscoveryNode {
    pub fn evaluation(&self) -> &EvaluationSpec {
        match self {
            Self::Derivation(spec) | Self::Aggregate(spec) | Self::Decision(spec) => spec,
            Self::Observation { evaluation, .. } => evaluation,
        }
    }

    pub fn cacheable(&self) -> bool {
        !matches!(self, Self::Observation { .. })
    }

    pub fn key(&self) -> ExperimentKey {
        match self {
            Self::Derivation(spec) => spec.semantic_key(b"derivation"),
            Self::Aggregate(spec) => spec.semantic_key(b"aggregate"),
            Self::Decision(spec) => spec.semantic_key(b"decision"),
            // The sample id makes an observation's identity explicit, but the
            // registry still refuses to cache it. Identity is provenance, not
            // permission to collapse independent evidence.
            Self::Observation { sample, evaluation } => {
                let base = evaluation.semantic_key(b"observation").0;
                ExperimentKey(hash_fields(
                    b"soma.discovery.observation.v1",
                    &[&base, &sample.to_le_bytes()],
                ))
            }
        }
    }

    pub fn digest(&self) -> NodeDigest {
        NodeDigest(self.key().0)
    }
}
