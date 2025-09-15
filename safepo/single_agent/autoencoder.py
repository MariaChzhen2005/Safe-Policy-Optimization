"""
    Conditional Autoencoder implementation.
"""
import torch
import torch.nn as nn
import numpy as np
from pypower.newtonpf import newtonpf
from pypower.ppoption import ppoption
from scipy.sparse import csr_matrix
import os

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalConstraintAwareAutoencoder(nn.Module):
    def __init__(self, action_dim, state_dim, latent_dim=None, hidden_dim=64,
                 num_decoders=1, latent_geom="hypersphere",
                 norm_params_path='ieee37bus/controller_inputs/normalization_params.npz',
                 ieee37_model_instance_in=None):
        """
        Conditional Autoencoder that encodes/decodes actions conditioned on states.
        
        Args:
            action_dim: Dimension of the action space
            state_dim: Dimension of the state/observation space
            latent_dim: Dimension of latent space (defaults to action_dim if None)
            hidden_dim: Hidden layer dimension
            num_decoders: Number of decoder experts
            latent_geom: Geometry of latent space ("hypersphere" or "hypercube")
            norm_params_path: Path to normalization parameters
            ieee37_model_instance_in: IEEE37 model instance for feasibility checking
        """
        super(ConditionalConstraintAwareAutoencoder, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim if latent_dim is not None else action_dim
        self.hidden_dim = hidden_dim
        self.num_decoders = num_decoders
        self.latent_geom = latent_geom

        # Latent convex set radius
        self.latent_radius = 0.5
        
        # Load normalization parameters and IEEE37 model instance for IEEE37
        self.norm_params_path = norm_params_path
        self.power_system_model = ieee37_model_instance_in
        self._n_buses_model = None
        self._gen_idx = None
        if ieee37_model_instance_in is not None:
            try:
                self._n_buses_model = int(ieee37_model_instance_in.n)
                self._gen_idx = np.array(ieee37_model_instance_in.gen_idx, dtype=int)
            except Exception:
                pass

        # Conditional Encoder: takes action and state, outputs latent
        self.encoder = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.Tanh()  # Constrains latent space to [-1, 1] hypercube
        )

        # Conditional Decoders: take latent and state, output action
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim + state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_decoders)
        ])
        
        # Gating network - determines weights for each decoder
        self.gating_network = nn.Sequential(
            nn.Linear(self.latent_dim + state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_decoders),
            nn.Softmax(dim=-1)  # Ensures weights sum to 1
        )
        
        # Feasibility predictor: takes state and action, predicts feasibility
        self.feasibility_predictor_nn = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode(self, action, state):
        """
        Encode action conditioned on state.
        Args:
            action: Action tensor (batch_size, action_dim)
            state: State tensor (batch_size, state_dim)
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        x = torch.cat([action, state], dim=1)
        return self.encoder(x)

    def decode(self, z, state):
        """
        Decode latent representations conditioned on state using mixture of experts.
        Args:
            z: Latent representations (batch_size, latent_dim)
            state: State tensor (batch_size, state_dim)
        Returns:
            Decoded actions (batch_size, action_dim)
        """
        z_state = torch.cat([z, state], dim=1)
        gate_weights = self.gating_network(z_state)
        
        decoder_outputs = []
        for decoder in self.decoders:
            output = decoder(z_state)
            decoder_outputs.append(output)
        
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        gate_weights = gate_weights.t().unsqueeze(-1)
        # weighted sum of decoder outputs
        mixed_output = (decoder_outputs * gate_weights).sum(dim=0)
        return mixed_output
    
    def decode_with_details(self, z, state):
        """
        Decode with detailed information about each expert's contribution.
        
        Args:
            z: Latent representations (batch_size, latent_dim)
            state: State tensor (batch_size, state_dim)
        Returns:
            mixed_output: Final decoded output
            decoder_outputs: List of outputs from each decoder
            gate_weights: Weights assigned to each decoder
        """
        z_state = torch.cat([z, state], dim=1)
        gate_weights = self.gating_network(z_state)
        
        decoder_outputs = []
        for decoder in self.decoders:
            output = decoder(z_state)
            decoder_outputs.append(output)
        
        decoder_outputs_stacked = torch.stack(decoder_outputs, dim=0)
        gate_weights_reshaped = gate_weights.t().unsqueeze(-1)
        mixed_output = (decoder_outputs_stacked * gate_weights_reshaped).sum(dim=0)
        
        return mixed_output, decoder_outputs, gate_weights

    def forward(self, action, state):
        """
        Forward pass through the conditional autoencoder
        Args:
            action: Action tensor (batch_size, action_dim)
            state: State tensor (batch_size, state_dim)
        Returns:
            action_recon: Reconstructed action
            z: Latent representation
        """
        z = self.encode(action, state)
        action_recon = self.decode(z, state)
        return action_recon, z
    
    def predict_feasibility_with_nn(self, action, state):
        """
        Predicts feasibility using the trained neural network predictor.
        Args:
            action: Action tensor (batch_size, action_dim)
            state: State tensor (batch_size, state_dim)
        Returns:
            Feasibility logits (batch_size, 1)
        """
        x = torch.cat([state, action], dim=1)
        return self.feasibility_predictor_nn(x)
    
    def verify_feasibility(self, action_batch, state_batch, shape_name):
        """
        Verifies feasibility of state-action pairs using ground-truth function.
        Args:
            action_batch: Batch of actions
            state_batch: Batch of states
            shape_name: Name of the shape/problem
        Returns:
            torch.Tensor: (batch_size, 1) with 1.0 for feasible and 0.0 for infeasible
        """
        if shape_name == 'ieee37bus':
            # Reconstruct full state-action for IEEE37 verification
            x_batch = torch.cat([state_batch, action_batch], dim=1)
            return self.verify_feasibility_with_newtonpf(x_batch)
        else:
            # For other shapes, concatenate state and action
            if isinstance(action_batch, torch.Tensor):
                x_batch = torch.cat([state_batch, action_batch], dim=1)
                x_np = x_batch.detach().cpu().numpy()
            else:
                x_np = np.concatenate([state_batch, action_batch], axis=1)
            
            feasible_mask = data_generation.check_feasibility(x_np, shape_name)
            feasibility_scores = torch.tensor(feasible_mask, dtype=torch.float32, device=device).unsqueeze(1)
            return feasibility_scores
    
    def verify_feasibility_with_newtonpf(self, x_normalized_batch):
        """
        Verifies feasibility using the full non-linear power flow (Newton PF).
        Input should be normalized state-action pairs.
        Returns: Tensor (B, 1) with {0.0, 1.0}.
        """
        if not os.path.exists(self.norm_params_path):
            raise FileNotFoundError(f"Normalization parameters file not found at '{self.norm_params_path}'.")
        norm = np.load(self.norm_params_path)
        mean = norm['mean'].astype(np.float32)
        std = norm['std'].astype(np.float32)
        gen_idx = norm['gen_idx'].astype(int) if 'gen_idx' in norm.files else None
        P_load_ref = norm['P_load_ref'].astype(np.float32) if 'P_load_ref' in norm.files else None
        Q_load_ref = norm['Q_load_ref'].astype(np.float32) if 'Q_load_ref' in norm.files else None
        n_buses_norm = int(norm['n_buses']) if 'n_buses' in norm.files else None

        if isinstance(x_normalized_batch, torch.Tensor):
            x_np = x_normalized_batch.detach().cpu().numpy().astype(np.float32)
            to_torch = True
            dev = x_normalized_batch.device
        else:
            x_np = np.asarray(x_normalized_batch, dtype=np.float32)
            to_torch = False
            dev = None

        B, D = x_np.shape
        ps = self.power_system_model
        if ps is None:
            raise ValueError("power_system_model is required for IEEE37 feasibility verification.")

        # Identify format
        mode = None
        if n_buses_norm is not None and D == 2 * n_buses_norm:
            mode = "full_bus"
        elif gen_idx is not None and D == 2 * len(gen_idx):
            mode = "inverter_only"
        else:
            # Fallback: try model attributes
            try:
                if D == 2 * ps.n:
                    mode = "full_bus"
                elif gen_idx is not None and D == 2 * len(gen_idx):
                    mode = "inverter_only"
            except Exception:
                pass
        if mode is None:
            raise ValueError(f"Unsupported input dimension D={D}. Expected 2*n_buses or 2*n_inverters.")

        # Denormalize to physical per-unit
        if mean.shape[0] != D or std.shape[0] != D:
            raise ValueError(f"Normalization shapes do not match input: mean/std have {mean.shape[0]}, expected {D}.")
        x_denorm = x_np * std[None, :] + mean[None, :]

        feas = np.zeros((B, 1), dtype=np.float32)

        if mode == "full_bus":
            n_b = D // 2
            for i in range(B):
                P_net = x_denorm[i, :n_b]
                Q_net = x_denorm[i, n_b:]
                S = P_net + 1j * Q_net
                ok, _, _ = ps.is_point_feasible(S, strict_check=True)
                feas[i, 0] = 1.0 if ok else 0.0
        else:
            if gen_idx is None or P_load_ref is None or Q_load_ref is None:
                # Use averages if reference not saved
                P_load_ref = ps.P_l.mean(axis=0).astype(np.float32)
                Q_load_ref = ps.Q_l.mean(axis=0).astype(np.float32)
                gen_idx = ps.gen_idx

            n_inv = len(gen_idx)
            for i in range(B):
                P_inv = x_denorm[i, :n_inv]
                Q_inv = x_denorm[i, n_inv:]
                P_net = -P_load_ref.copy()
                Q_net = -Q_load_ref.copy()
                P_net[gen_idx] += P_inv
                Q_net[gen_idx] += Q_inv
                S = P_net + 1j * Q_net
                ok, _, _ = ps.is_point_feasible(S, strict_check=True)
                feas[i, 0] = 1.0 if ok else 0.0

        if to_torch:
            return torch.tensor(feas, device=dev, dtype=torch.float32)
        else:
            return feas
    
    def project_action(self, action_batch, state_batch):
        """
        Project actions to feasible set conditioned on states.
        Args:
            action_batch: Proposed actions (batch_size, action_dim)
            state_batch: Current states (batch_size, state_dim)
        Returns:
            Projected feasible actions (batch_size, action_dim)
        """
        z = self.encode(action_batch, state_batch)
        z_norm = torch.norm(z, dim=1, keepdim=True)
        
        if self.latent_geom == "hypersphere":
            z_projected = torch.where(z_norm > self.latent_radius, 
                                     z * (self.latent_radius / z_norm), z)
        elif self.latent_geom == "hypercube":
            z_projected = torch.clamp(z, min=-self.latent_radius, max=self.latent_radius)
        else:
            z_projected = z
            
        return self.decode(z_projected, state_batch)

def geometric_regularization_loss(model, z_batch, state_batch, alpha=1.0):
    """
    Encourage uniform Jacobian determinants across latent space.
    Args:
        model: The conditional autoencoder model
        z_batch: Batch of latent vectors
        state_batch: Batch of state vectors
        alpha: Weight for the regularization
    """
    if not z_batch.requires_grad:
        z_batch.requires_grad_(True)
    batch_size = z_batch.size(0)
    log_det_values = []
    num_samples_to_process = min(batch_size, 32)

    for i in range(num_samples_to_process):
        z_sample_i = z_batch[i:i+1]
        state_sample_i = state_batch[i:i+1]
        action_decoded_i = model.decode(z_sample_i, state_sample_i)

        jacobian_rows = []
        for j in range(action_decoded_i.size(1)):
            grad_j = torch.autograd.grad(
                outputs=action_decoded_i[0, j],
                inputs=z_sample_i,
                grad_outputs=torch.ones_like(action_decoded_i[0, j]),
                retain_graph=True,
                create_graph=True
            )[0]
            jacobian_rows.append(grad_j)
        jacobian_i = torch.stack(jacobian_rows, dim=0)
        jacobian_i = jacobian_i.squeeze(1)

        matrix_for_det = jacobian_i @ jacobian_i.T + 1e-6 * torch.eye(jacobian_i.size(0), device=jacobian_i.device)
        det = torch.det(matrix_for_det)
        # Guard against non-finite or non-positive determinants
        if torch.isfinite(det) and det > 0:
            # Clamp to avoid extreme values causing inf during log/variance
            det_clamped = torch.clamp(det, min=1e-12, max=1e12)
            log_det_values.append(torch.log(det_clamped))

    if len(log_det_values) == 0:
        return torch.tensor(0.0, device=z_batch.device, dtype=z_batch.dtype)

    log_jacobian_dets = torch.stack(log_det_values)
    log_jacobian_dets = torch.nan_to_num(log_jacobian_dets, nan=0.0, posinf=0.0, neginf=0.0)
    variance_penalty = torch.var(log_jacobian_dets)

    return alpha * variance_penalty

def compute_density_loss(z_batch, eps=1e-6):
    """
    Compute density loss to encourage uniform distribution in latent space
    """
    batch_size = z_batch.size(0)
    z_expanded = z_batch.unsqueeze(1).repeat(1, batch_size, 1)
    z_expanded_t = z_batch.unsqueeze(0).repeat(batch_size, 1, 1)
    dist_matrix = torch.norm(z_expanded - z_expanded_t, dim=2)

    inf_mask = torch.eye(batch_size, device=device) * 10**12
    dist_matrix = dist_matrix + inf_mask
    min_dists = torch.min(dist_matrix, dim=1)[0]

    # add epsilon for numerical stability
    density_loss = -torch.mean(torch.log(min_dists + eps))

    return density_loss