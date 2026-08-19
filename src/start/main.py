import os
import sys
from typing import Any


def execute_pre_flight_summary_card(config: dict[str, Any]) -> None:
    """Renders a meticulous pre-flight card layout before waking the agent committee."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("╭─ Pre-Flight Configuration Review ──────────────────────────────────────────╮")
    print(f"│  Workflow Mode     {config['workflow_mode']:55} │")
    print(f"│  Agent Mode        {config.get('agent_mode', 'deterministic'):55} │")
    print(f"│  LLM Provider      {config.get('llm_provider', 'none'):55} │")
    if config.get("llm_model"):
        print(f"│  LLM Model         {config['llm_model']:55} │")
    print(f"│  Selected Model    {config['model']:55} │")
    if config['workflow_mode'] == "Deep Learning Suite":
        print(f"│  Activation        {config.get('activation', 'relu'):55} │")
        if "Regression" in config['problem_frame']:
            out_act, loss_fn = "Linear", "Mean Squared Error"
            out_layer = "1 output"
        elif "Multi-Class" in config['problem_frame']:
            out_act, loss_fn = "Softmax", "Categorical Cross Entropy"
            out_layer = f"{config.get('n_classes', 3)} outputs"
        else:
            out_act, loss_fn = "Sigmoid", "Binary Cross Entropy"
            out_layer = "1 output"
        print(f"│  Output Layer      {out_layer:55} │")
        print(f"│  Output Activation {out_act:55} │")
        print(f"│  Loss Function     {loss_fn:55} │")
    print(f"│  Dataset Vector    {config['dataset_vector']:55} │")
    print(f"│  Public Source     {config['public_source']:55} │")
    print(f"│  Data Volumetrics  {config['volumetrics']:55} │")
    print(f"│  Problem Frame     {config['problem_frame']:55} │")
    print(f"│  Target Column     {config['target_column']:55} │")
    print(f"│  Class Dist.       {config['class_distribution']:55} │")
    
    strat_status = "Enabled" if config["stratify"] else "Disabled"
    cw_status = "Balanced (sample weight)" if config["class_weight"] else "Disabled"
    print(f"│  Stratification    {strat_status:55} │")
    print(f"│  Class Weighting   {cw_status:55} │")
    
    train_pct = int(config["split_proportions"][0] * 100)
    test_pct = int(config["split_proportions"][1] * 100)
    oos_pct = int(config["split_proportions"][2] * 100)
    gen_str = f"Split Matrix [Train: {train_pct}% | Test: {test_pct}% | OOS: {oos_pct}%]"
    print(f"│  Generalization    {gen_str:55} │")
    print("│  Numerical Guard   Epsilon Bounds Active (Denom Min: 1e-15)               │")
    print("╰────────────────────────────────────────────────────────────────────────────╯")
    choice = input("Proceed to execute AI Engineering Review Committee? [Y/n]: ").strip().lower()
    if choice == 'n':
        print("Execution halted by engineering override command.")
        sys.exit(0)

def run_interactive_setup_wizard() -> dict[str, Any]:
    """Interactive wizard parsing user inputs into a structured configuration state."""
    print("=== StART Model Review — Interactive Setup Wizard ===")
    
    # 0. AI Reviewer Agent Backend Selection (LLM Mode)
    print("\nSelect AI Reviewer Agent Backend (LLM Mode):")
    print("  [1] None (Deterministic Rule-Based / Local Engines) (default)")
    print("  [2] Enterprise LLM Gateway (Firm Environment)")
    print("  [3] Public LLM Providers (OpenAI, Anthropic, Gemini, DeepSeek, Groq, etc. - Paid API Keys required)")
    backend_choice = input("Select backend [default: 1]: ").strip() or "1"
    
    agent_mode = "deterministic"
    llm_provider = "none"
    llm_model = None
    
    if backend_choice == "2":
        agent_mode = "llm"
        llm_provider = "enterprise_llm_gateway"
    elif backend_choice == "3":
        agent_mode = "llm"
        print("\nSelect Public LLM Provider:")
        print("  [1] OpenAI (default)")
        print("  [2] Anthropic")
        print("  [3] Gemini")
        print("  [4] DeepSeek")
        print("  [5] Grok")
        provider_choice = input("Select provider [default: 1]: ").strip() or "1"
        provider_map = {
            "1": "openai",
            "2": "anthropic",
            "3": "gemini",
            "4": "deepseek",
            "5": "grok"
        }
        llm_provider = provider_map.get(provider_choice, "openai")
        
        # Securely prompt for key immediately if needed
        import sys

        from start.providers.keys import ensure_provider_key, key_required
        if key_required(llm_provider) and sys.stdin.isatty():
            status = ensure_provider_key(llm_provider, prompt_for_key=True, interactive=True)
            if not status.ok:
                print(f"\n[Warning] API key for '{llm_provider}' is missing or empty.")
                print("Degrading to deterministic rule-based backend for this session.")
                agent_mode = "deterministic"
                llm_provider = "none"

        llm_model = None
        if llm_provider != "none" and llm_provider in ("openai", "anthropic"):
            from start.providers.model_discovery import RealProviderModelDiscovery
            discovery = RealProviderModelDiscovery()
            print(f"\nQuerying available models for {llm_provider}...")
            available_models = discovery.list_models(llm_provider)
            if available_models:
                print(f"\nAvailable models for {llm_provider}:")
                for idx, mid in enumerate(available_models, 1):
                    print(f"  [{idx}] {mid}")
                while True:
                    model_choice = input(f"Select model (1-{len(available_models)}): ").strip()
                    try:
                        sel_idx = int(model_choice) - 1
                        if 0 <= sel_idx < len(available_models):
                            llm_model = available_models[sel_idx]
                            break
                    except ValueError:
                        pass
                    if model_choice in available_models:
                        llm_model = model_choice
                        break
                    print("Invalid selection. Please choose a valid index or type the model ID exactly.")
            else:
                print(f"\n[Warning] No models returned by {llm_provider} API automatically.")
                while True:
                    manual_model = input("Enter the model ID manually: ").strip()
                    if manual_model:
                        llm_model = manual_model
                        break
                    print("Model ID cannot be empty. Please enter a valid model ID.")

    objective = ""
    clarification = ""
    if agent_mode == "llm":
        print("\nEnter Business Context & Reviewer Clarification:")
        objective = input("Business context: ").strip()
        clarification = input("Reviewer clarification: ").strip()

    # 1. Branch Selection
    print("\nSelect Evaluation Suite Branch:")
    print("  [1] Propensity Suite (Traditional Tree-Based ML: Random Forest, CatBoost, XGBoost)")
    print("  [2] Deep Learning Suite (Neural Network Architecture Ops: Wide & Deep, MLP)")
    suite_choice = input("Select branch [default: 1]: ").strip() or "1"
    workflow_mode = "Deep Learning Suite" if suite_choice == "2" else "Propensity Suite"

    activation = "relu"

    # 1.5. Model Selection
    if workflow_mode == "Propensity Suite":
        print("\nSelect Propensity Model:")
        print("  [1] Random Forest (default)")
        print("  [2] CatBoost")
        print("  [3] XGBoost")
        print("  [4] LightGBM")
        print("  [5] Distributed Random Forest")
        print("  [6] Extra Trees")
        print("  [7] Random Rotation Forest")
        model_choice = input("Select model [default: 1]: ").strip() or "1"
        model_map = {
            "1": "random_forest",
            "2": "catboost",
            "3": "xgboost",
            "4": "lightgbm",
            "5": "distributed_random_forest",
            "6": "extra_trees",
            "7": "random_rotation_forest"
        }
        model = model_map.get(model_choice, "random_forest")
    else:
        print("\nSelect Neural Network Architecture:")
        print("  [1] MLP (default)")
        print("  [2] RNN")
        print("  [3] LSTM")
        print("  [4] CNN")
        print("  [5] GRU")
        print("  [6] Bi-LSTM")
        print("  [7] Graph Neural Network (GCN/GAT for relational networks and AML tracking)")
        print("  [8] Wide & Deep / Deep & Cross Network (DCN for automated tabular feature interactions)")
        model_choice = input("Select architecture [default: 1]: ").strip() or "1"
        model_map = {
            "1": "mlp",
            "2": "rnn",
            "3": "lstm",
            "4": "cnn",
            "5": "gru",
            "6": "bi_lstm",
            "7": "gnn",
            "8": "dcn"
        }
        model = model_map.get(model_choice, "mlp")

        # 1.6 Dedicated activation selection layer immediately after architecture selection
        print("\nSelect Activation Function:")
        print("  [1] ReLU (default)")
        print("  [2] LeakyReLU")
        print("  [3] GELU")
        print("  [4] Swish")
        print("  [5] Mish")
        print("  [6] ELU")
        print("  [7] SELU")
        print("  [8] Tanh")
        print("  [9] Sigmoid")
        print("  [10] Softplus")
        act_choice = input("Select activation [default: 1]: ").strip() or "1"
        act_map = {
            "1": "relu",
            "2": "leaky_relu",
            "3": "gelu",
            "4": "swish",
            "5": "mish",
            "6": "elu",
            "7": "selu",
            "8": "tanh",
            "9": "sigmoid",
            "10": "softplus"
        }
        activation = act_map.get(act_choice, "relu")

    # 2. Domain Dataset Ingress Selector
    from start.data.selection import WIZARD_OPTIONS, resolve_wizard_choice

    print("\nSelect Dataset Source:")
    for key, text in WIZARD_OPTIONS:
        print(f"  [{key}] {text}")
    ds_choice = input("Select dataset option [default: 1]: ").strip() or "1"

    selection = resolve_wizard_choice(ds_choice, seed=42)

    errors = selection.consistency_errors()
    if errors:
        print("\n[Error] Dataset provenance is inconsistent:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    df = selection.frame
    default_target = selection.target_column or "inferred_by_discovery"
    dataset_vector = selection.display_name
    public_source = selection.source_reference
    dataset_path = selection.source_path or ""
    dataset_source = selection.provenance_dict()

    volumetrics = f"{df.shape[0]} Rows x {df.shape[1]} Dimensions"

    # 3. Task & Segmentation Selection
    task_type = input("\nSelect Problem Task Type [1=Classification / 2=Regression]: ").strip() or "1"
    if task_type == "2":
        problem_frame = "Regression ──> Continuous Numbers"
    else:
        seg_type = input("Select Segmentation Mode [1=Binary / 2=Multi-Class]: ").strip() or "1"
        problem_frame = "Classification ──> Binary Segment" if seg_type == "1" else "Classification ──> Multi-Class Segmentation"

    # 4. Target Column Selection & Verification
    target_input = input(f"\nEnter target column name (blank = auto-lock fallback choice '{default_target}'): ").strip()
    target_column = target_input or default_target

    # Rename target column in demo dataset if needed
    if not dataset_path and target_column != default_target and default_target in df.columns:
        df = df.rename(columns={default_target: target_column})

    # Target column validation loop
    if target_column not in df.columns:
        print(f"\n[Warning] Target column '{target_column}' not found in the dataset.")
        print("Please choose a valid target column from the list below:")
        cols = list(df.columns)
        for idx, col in enumerate(cols):
            print(f"  [{idx + 1}] {col}")
        while True:
            col_choice = input(f"Select column index [1-{len(cols)}]: ").strip()
            if not col_choice:
                target_column = cols[-1]
                break
            if col_choice.isdigit():
                idx = int(col_choice) - 1
                if 0 <= idx < len(cols):
                    target_column = cols[idx]
                    break
            print("Invalid selection. Please enter a valid number.")

    # Reconcile target cardinality & task segmentation mode early
    n_classes = 1
    if task_type == "1":
        nunique = df[target_column].dropna().nunique()
        n_classes = nunique
        if seg_type == "1" and nunique > 2:
            print(f"\n[Warning] Target column '{target_column}' has {nunique} unique values, "
                  f"which requires Multi-Class classification, but you selected Binary Segmentation.")
            choice = input("Confirm: Switch to Multi-Class Segmentation? [Y/n]: ").strip().lower()
            if choice in ("", "y", "yes"):
                seg_type = "2"
                problem_frame = "Classification ──> Multi-Class Segmentation"
                print("Switched to Multi-Class classification.")
            else:
                print("Execution aborted: Selected binary classification is incompatible with multiclass target.")
                sys.exit(1)

    if task_type == "2":
        resolved_task_type = "regression"
    else:
        resolved_task_type = "multiclass_classification" if seg_type == "2" else "binary_classification"

    # 5. Split and Weighting Choices
    sample_weight = False
    split_strategy = "stratified"
    
    if task_type == "2":
        # Regression
        print("\nSelect Split Strategy for Regression:")
        print("  [1] Random (default)")
        print("  [2] Time-Series Split")
        print("  [3] Group-Based")
        print("  [4] Binned Target Split (disclosed target stratification)")
        split_choice = input("Select split strategy [default: 1]: ").strip() or "1"
        split_map = {
            "1": "random",
            "2": "time_based",
            "3": "group",
            "4": "stratified"
        }
        split_strategy = split_map.get(split_choice, "random")
        stratify = (split_strategy == "stratified")
        class_weight = None
        
        print("\nSelect Sample Weighting Logic for Regression:")
        print("  [1] No weighting (default)")
        print("  [2] Inverse Variance Weighting")
        print("  [3] Temporal Decay Weighting")
        print("  [4] Custom Weighting Logic")
        sw_choice = input("Select weighting logic [default: 1]: ").strip() or "1"
        if sw_choice in ("2", "3", "4"):
            justification = input("Enter explicit justification for the selected weighting logic (blank to disable weighting): ").strip()
            if justification:
                sample_weight = True
                print(f"Sample weights enabled with justification: {justification}")
            else:
                sample_weight = False
                print("No justification provided. Sample weights disabled.")
        else:
            sample_weight = False
    else:
        # Classification
        strat_choice = input("\nApply stratified splitting to training/test/OOS cohorts? [Y/n]: ").strip().lower()
        stratify = strat_choice not in ("n", "no")
        split_strategy = "stratified" if stratify else "random"
        
        cw_choice = input("\nApply class-weight balancing (highly recommended for imbalanced datasets / anomaly detection)? [y/N]: ").strip().lower()
        class_weight = "balanced" if cw_choice in ("y", "yes") else None

    # Print user-friendly notes/warnings about sequential lookahead leakage
    if split_strategy == "random":
        print("\n[Note] Random split strategy does not preserve temporal ordering.")
        print("       If your dataset has sequential dependencies (e.g., lag features), this will cause")
        print("       severe lookahead data leakage, leading to optimistic train scores and OOS collapse.")
    elif split_strategy == "time_based":
        print("\n[Note] Time-Series split strategy preserves temporal ordering.")
        print("       This guards against lookahead data leakage by training only on historical data.")

    # 7. Split Proportions Choice
    split_choice = input("\nEnter train/test/OOS split proportions (must sum to 100, e.g., 60/20/20) [default: 60/20/20]: ").strip()
    split_props = (0.6, 0.2, 0.2)
    if split_choice:
        try:
            parts = [float(p.strip()) for p in split_choice.split("/")]
            if len(parts) == 3 and sum(parts) == 100:
                split_props = (parts[0]/100.0, parts[1]/100.0, parts[2]/100.0)
            else:
                print("Proportions must be 3 numbers summing to 100. Using default 60/20/20.")
        except Exception:
            print("Invalid format. Using default 60/20/20.")

    # 8. User-Controlled Hyperparameter Tuning configuration
    tuning_needed_choice = input("\nEnable hyperparameter tuning? [y/N]: ").strip().lower()
    tuning_needed = tuning_needed_choice in ("y", "yes")
    
    tuning_strategy = "none"
    tuning_trials = 0
    validation_scheme = "holdout"
    k_folds = 3
    custom_tuning_params = {}
    
    if tuning_needed:
        print("\nSelect Tuning Method:")
        print("  [1] Bounded Random Search (default)")
        print("  [2] Grid Search")
        print("  [3] Optuna Search (if available)")
        method_choice = input("Select method [default: 1]: ").strip() or "1"
        method_map = {
            "1": "bounded_random_search",
            "2": "grid_search",
            "3": "optuna_if_available"
        }
        tuning_strategy = method_map.get(method_choice, "bounded_random_search")
        
        trials_choice = input("\nEnter number of trials [default: 5]: ").strip()
        tuning_trials = int(trials_choice) if trials_choice.isdigit() and int(trials_choice) > 0 else 5
        
        print("\nSelect Validation Scheme:")
        print("  [1] Holdout (default)")
        print("  [2] K-Fold")
        print("  [3] Time-Series Split")
        val_choice = input("Select validation scheme [default: 1]: ").strip() or "1"
        val_map = {
            "1": "holdout",
            "2": "k_fold",
            "3": "time_series_split"
        }
        validation_scheme = val_map.get(val_choice, "holdout")
        
        if validation_scheme == "k_fold":
            print("\nSelect Number of Folds:")
            print("  [1] 3 (default)")
            print("  [2] 5")
            print("  [3] 7")
            k_choice = input("Select folds [default: 1]: ").strip() or "1"
            k_map = {"1": 3, "2": 5, "3": 7}
            k_folds = k_map.get(k_choice, 3)
            
        # Hyperparameters prompting by architecture/model:
        # MLP / Wide & Deep / DCN
        if model in ("mlp", "wide_deep", "dcn"):
            h_dims_raw = input("\nEnter hidden dimensions (comma separated integers) [default: 128, 64]: ").strip()
            if h_dims_raw:
                custom_tuning_params["hidden_dims"] = [int(x.strip()) for x in h_dims_raw.split(",")]
            
            lr_raw = input("\nEnter learning rate [default: 0.003]: ").strip()
            if lr_raw:
                custom_tuning_params["learning_rate"] = float(lr_raw)
            
            dr_raw = input("\nEnter dropout [default: 0.1]: ").strip()
            if dr_raw:
                custom_tuning_params["dropout"] = float(dr_raw)
            
            wd_raw = input("\nEnter weight decay [default: 1e-4]: ").strip()
            if wd_raw:
                custom_tuning_params["weight_decay"] = float(wd_raw)
            
            bs_raw = input("\nEnter batch size [default: 64]: ").strip()
            if bs_raw:
                custom_tuning_params["batch_size"] = int(bs_raw)
            
            ep_raw = input("\nEnter epochs [default: 50]: ").strip()
            if ep_raw:
                custom_tuning_params["epochs"] = int(ep_raw)
            
        # RNN / LSTM / GRU / Bi-LSTM
        elif model in ("rnn", "lstm", "gru", "bi_lstm"):
            hs_raw = input("\nEnter hidden size [default: 64]: ").strip()
            if hs_raw:
                custom_tuning_params["hidden_size"] = int(hs_raw)
            
            nl_raw = input("\nEnter number of layers [default: 1]: ").strip()
            if nl_raw:
                custom_tuning_params["num_layers"] = int(nl_raw)
            
            lr_raw = input("\nEnter learning rate [default: 0.003]: ").strip()
            if lr_raw:
                custom_tuning_params["learning_rate"] = float(lr_raw)
            
            dr_raw = input("\nEnter dropout [default: 0.1]: ").strip()
            if dr_raw:
                custom_tuning_params["dropout"] = float(dr_raw)
            
            seq_raw = input("\nEnter sequence length [default: auto]: ").strip()
            if seq_raw:
                custom_tuning_params["sequence_length"] = int(seq_raw) if seq_raw.isdigit() else "auto"
            
            bs_raw = input("\nEnter batch size [default: 64]: ").strip()
            if bs_raw:
                custom_tuning_params["batch_size"] = int(bs_raw)
            
        # CNN
        elif model == "cnn":
            nf_raw = input("\nEnter number of filters [default: 32]: ").strip()
            if nf_raw:
                custom_tuning_params["num_filters"] = int(nf_raw)
            
            ks_raw = input("\nEnter kernel size [default: 3]: ").strip()
            if ks_raw:
                custom_tuning_params["kernel_size"] = int(ks_raw)
            
            lr_raw = input("\nEnter learning rate [default: 0.003]: ").strip()
            if lr_raw:
                custom_tuning_params["learning_rate"] = float(lr_raw)
            
            dr_raw = input("\nEnter dropout [default: 0.1]: ").strip()
            if dr_raw:
                custom_tuning_params["dropout"] = float(dr_raw)
            
            bs_raw = input("\nEnter batch size [default: 64]: ").strip()
            if bs_raw:
                custom_tuning_params["batch_size"] = int(bs_raw)
            
            ep_raw = input("\nEnter epochs [default: 50]: ").strip()
            if ep_raw:
                custom_tuning_params["epochs"] = int(ep_raw)
            
        # GNN
        elif model == "gnn":
            hc_raw = input("\nEnter hidden channels [default: 64]: ").strip()
            if hc_raw:
                custom_tuning_params["hidden_channels"] = int(hc_raw)
            
            nl_raw = input("\nEnter number of layers [default: 2]: ").strip()
            if nl_raw:
                custom_tuning_params["num_layers"] = int(nl_raw)
            
            lr_raw = input("\nEnter learning rate [default: 0.003]: ").strip()
            if lr_raw:
                custom_tuning_params["learning_rate"] = float(lr_raw)
            
            dr_raw = input("\nEnter dropout [default: 0.1]: ").strip()
            if dr_raw:
                custom_tuning_params["dropout"] = float(dr_raw)
            
            agg_raw = input("\nEnter aggregation method (mean/sum/max) [default: mean]: ").strip().lower()
            if agg_raw:
                custom_tuning_params["aggregation"] = agg_raw if agg_raw in ("mean", "sum", "max") else "mean"
            
            ep_raw = input("\nEnter epochs [default: 50]: ").strip()
            if ep_raw:
                custom_tuning_params["epochs"] = int(ep_raw)
            
        # Random Forest (Propensity Suite)
        elif model in ("random_forest", "distributed_random_forest", "extra_trees", "random_rotation_forest"):
            ne_raw = input("\nEnter number of estimators [default: 300]: ").strip()
            if ne_raw:
                custom_tuning_params["n_estimators"] = int(ne_raw)
            
            md_raw = input("\nEnter max depth (integer or auto) [default: auto]: ").strip()
            if md_raw:
                custom_tuning_params["max_depth"] = int(md_raw) if md_raw.isdigit() else None
            
            msl_raw = input("\nEnter min samples leaf [default: 5]: ").strip()
            if msl_raw:
                custom_tuning_params["min_samples_leaf"] = int(msl_raw)
            
            mf_raw = input("\nEnter max features (sqrt/log2/None) [default: sqrt]: ").strip()
            if mf_raw:
                custom_tuning_params["max_features"] = mf_raw if mf_raw in ("sqrt", "log2", "none") else "sqrt"
                if custom_tuning_params["max_features"] == "none":
                    custom_tuning_params["max_features"] = None
                
            custom_tuning_params["class_weight"] = "balanced" if task_type == "1" else None
            custom_tuning_params["random_state"] = 42
            
        # XGBoost / LightGBM / CatBoost (Propensity Suite)
        elif model in ("xgboost", "lightgbm", "catboost"):
            ne_raw = input("\nEnter number of estimators/iterations [default: 300]: ").strip()
            if ne_raw:
                custom_tuning_params["n_estimators"] = int(ne_raw)
            
            md_raw = input("\nEnter max depth [default: 4]: ").strip()
            if md_raw:
                custom_tuning_params["max_depth"] = int(md_raw)
            
            lr_raw = input("\nEnter learning rate [default: 0.05]: ").strip()
            if lr_raw:
                custom_tuning_params["learning_rate"] = float(lr_raw)
            
            ss_raw = input("\nEnter subsample [default: 0.8]: ").strip()
            if ss_raw:
                custom_tuning_params["subsample"] = float(ss_raw)
            
            cbt_raw = input("\nEnter colsample by tree [default: 0.8]: ").strip()
            if cbt_raw:
                custom_tuning_params["colsample_bytree"] = float(cbt_raw)
            
            rl_raw = input("\nEnter reg lambda [default: 1.0]: ").strip()
            if rl_raw:
                custom_tuning_params["reg_lambda"] = float(rl_raw)

    # 8.5 Select Explainability Method
    print("\nSelect Explainability Method:")
    print("  [1] Integrated Gradients (default)")
    print("  [2] Gradient SHAP")
    print("  [3] Permutation Importance")
    exp_choice = input("Select explainability method [default: 1]: ").strip() or "1"
    exp_map = {
        "1": "integrated_gradients",
        "2": "gradient_shap",
        "3": "permutation"
    }
    explain_method = exp_map.get(exp_choice, "integrated_gradients")

    # Calculate class distribution percentages
    if task_type == "1":
        counts = df[target_column].value_counts(normalize=True) * 100
        class_dist_str = ", ".join([f"'{val}': {pct:.1f}%" for val, pct in counts.items()])
    else:
        class_dist_str = "N/A (Regression Task)"

    config_state = {
        "workflow_mode": workflow_mode,
        "model": model,
        "activation": activation,
        "agent_mode": agent_mode,
        "llm_provider": llm_provider,
        "dataset_vector": dataset_vector,
        "preset_key": ds_choice,
        "dataset_selection": selection,
        "dataset_provenance": selection.provenance_dict(),
        "dataset_source": dataset_source,
        "public_source": public_source,
        "volumetrics": volumetrics,
        "problem_frame": problem_frame,
        "target_column": target_column,
        "target_arrays": [target_column],
        "class_distribution": class_dist_str,
        "stratify": stratify,
        "class_weight": class_weight,
        "split_proportions": split_props,
        "split_strategy_name": split_strategy,
        "sample_weight": sample_weight,
        "tuning_strategy": tuning_strategy,
        "tuning_trials": tuning_trials,
        "validation_scheme": validation_scheme,
        "k_folds": k_folds,
        "explain_method": explain_method,
        "custom_tuning_params": custom_tuning_params,
        "objective": objective,
        "clarification": clarification,
        "task_type": resolved_task_type,
        "n_classes": n_classes,
        "llm_model": llm_model,
    }

    execute_pre_flight_summary_card(config_state)
    return config_state

if __name__ == "__main__":
    config = run_interactive_setup_wizard()
    print(f"\nConfiguration locked. Initializing pipeline runner targeting {config['workflow_mode']}...")
