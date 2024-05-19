import optuna
import subprocess

def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    n_layer = trial.suggest_int('n_layer', 6, 24)
    n_embd = trial.suggest_categorical('n_embd', [512, 768, 1024])
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

    # Create the command-line arguments for the training script
    args = [
        'python', 'train.py',
        f'--learning_rate={learning_rate}',
        f'--batch_size={batch_size}',
        f'--n_layer={n_layer}',
        f'--n_embd={n_embd}',
        f'--dropout={dropout}',
        '--eval_only',
    ]

    # Run the training script as a subprocess
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Parse the best validation loss from the training script output
    try:
        best_val_loss = float(stdout.decode().split('best val loss: ')[1].split(',')[0])
    except (IndexError, ValueError):
        print(f"Error parsing validation loss from stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
        best_val_loss = float('inf')

    return best_val_loss

# Create an Optuna study
study = optuna.create_study(direction='minimize')

# Optimize the hyperparameters
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and best validation loss
print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)
