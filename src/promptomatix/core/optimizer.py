"""
Core module for prompt optimization functionality.
"""

import dspy
import ast
import json
import sys
import threading
from typing import Dict, List, Type, Optional, Union, Tuple
from datetime import datetime
from dspy.evaluate import Evaluate
import nltk
import os
import logging
from pathlib import Path
import requests

from ..utils.parsing import parse_dict_strings
from ..utils.paths import OPTIMIZER_LOGS_DIR
from ..core.config import Config
from ..core.session import OptimizationSession
from ..metrics.metrics import MetricsManager
from .prompts import (
    generate_synthetic_data_prompt, 
    generate_synthetic_data_validation_prompt,
    generate_meta_prompt,
    generate_meta_prompt_7,
    validate_synthetic_data,
    generate_meta_prompt_2
)

# Setup module logger
logger = logging.getLogger(__name__)

class PromptOptimizer:
    """
    Handles the optimization of prompts using either DSPy or meta-prompt backend.
    
    Attributes:
        config (Config): Configuration for optimization
        lm: Language model instance
        optimized_prompt (str): Latest optimized prompt
        data_template (Dict): Template for data structure
        backend (str): Optimization backend ('dspy' or 'simple_meta_prompt')
    """
    
    def __init__(self, config: Config):
        """
        Initialize the optimizer with configuration.
        
        Args:
            config (Config): Configuration object containing optimization parameters
        """
        self.config = config
        self.lm = None
        self.llm_cost = 0
        # Initialize logger
        setup_optimizer_logger()
        self.logger = logger
        self.logger.info("PromptOptimizer initialized")
        
        # Set backend (default to DSPy for backward compatibility)
        self.backend = getattr(config, 'backend', 'simple_meta_prompt')
        if self.backend not in ['dspy', 'simple_meta_prompt']:
            raise ValueError(f"Unsupported backend: {self.backend}. Must be 'dspy' or 'simple_meta_prompt'")
        
        self.logger.info(f"Using backend: {self.backend}")

    def create_signature(self, name: str, input_fields: List[str], 
                        output_fields: List[str]) -> Type[dspy.Signature]:
        """
        Create a DSPy signature for the optimization task.
        
        Args:
            name (str): Name of the signature
            input_fields (List[str]): List of input field names
            output_fields (List[str]): List of output field names
            
        Returns:
            Type[dspy.Signature]: DSPy signature class
        """
        cleaned_name = name.strip('_').strip()
        
        # Parse fields if they're strings
        input_fields = self._parse_fields(input_fields)
        output_fields = self._parse_fields(output_fields)
        
        # Create signature class
        class_body = {
            '__annotations__': {},
            '__doc__': self.config.task
        }
        
        # Add input and output fields
        for field in input_fields:
            field = field.strip('"\'')
            class_body['__annotations__'][field] = str
            class_body[field] = dspy.InputField()
            
        for field in output_fields:
            field = field.strip('"\'')
            class_body['__annotations__'][field] = str
            class_body[field] = dspy.OutputField()
        
        return type(cleaned_name, (dspy.Signature,), class_body)
    
    def _parse_fields(self, fields: Union[List[str], str]) -> List[str]:
        """Parse field definitions from string or list."""
        if isinstance(fields, str):
            fields = fields.strip()
            if not (fields.startswith('[') and fields.endswith(']')):
                fields = f"[{fields}]"
            return ast.literal_eval(fields)
        return fields

    def generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic training data based on sample data in batches."""
        try:
            sample_data, sample_data_group = self._prepare_sample_data()
            template = {key: '...' for key in sample_data.keys()}

            # On average, 4 characters make up a token
            no_of_toks_in_sample_data = len(str(sample_data))/4
            
            # Calculate batch size based on token limits (assuming 16k token limit)
            max_samples_per_batch = min(50, max(1, int(8000 / no_of_toks_in_sample_data)))
            
            all_synthetic_data = []
            remaining_samples = self.config.synthetic_data_size
            max_retries = 3  # Maximum number of retries per batch
            validation_feedback = []  # Store feedback for failed attempts
            
            print(f"📊 Generating {self.config.synthetic_data_size} synthetic samples...")
            
            # Initialize LLM once for all batches
            with dspy.settings.context():
                try:
                    # Check if using Azure OpenAI
                    azure_key = os.getenv("AZURE_OPENAI_KEY")
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                    use_azure = azure_key and azure_endpoint and deployment_name
                    
                    if use_azure and self.config.config_model_name and self.config.config_model_name.startswith("azure/"):
                        # Use Azure OpenAI - DSPy will read from environment variables
                        tmp_lm = dspy.LM(
                            self.config.config_model_name,
                            max_tokens=self.config.config_max_tokens,
                            cache=False
                        )
                    else:
                        # Use standard provider
                        tmp_lm = dspy.LM(
                            self.config.config_model_name,
                            api_key=self.config.config_model_api_key,
                            api_base=self.config.config_model_api_base,
                            max_tokens=self.config.config_max_tokens,
                            cache=False
                        )
                    
                    batch_num = 1
                    while remaining_samples > 0:
                        batch_size = min(max_samples_per_batch, remaining_samples)
                        
                        valid_batch_data = []
                        retry_count = 0
                        
                        while len(valid_batch_data) < batch_size and retry_count < max_retries:
                            try:
                                # Include previous validation feedback in the prompt
                                feedback_section = ""
                                if validation_feedback:
                                    feedback_section = "\n### Previous Validation Feedback:\n" + "\n".join(
                                        f"- Attempt {i+1}: {feedback}" 
                                        for i, feedback in enumerate(validation_feedback[-3:])  # Show last 3 feedbacks
                                    )
                                
                                prompt = self._create_synthetic_data_prompt(
                                    sample_data_group, 
                                    template, 
                                    batch_size - len(valid_batch_data),
                                    feedback_section
                                )
                                
                                response = tmp_lm(prompt)[0]
                                response = self._clean_llm_response(response)
                                
                                try:
                                    batch_data = json.loads(response)
                                    # Validate each sample in the batch
                                    for sample in batch_data:
                                        is_valid, feedback = self._validate_synthetic_data(sample, self.config.task)
                                        if is_valid:
                                            valid_batch_data.append(sample)
                                            if len(valid_batch_data) >= batch_size:
                                                break
                                        else:
                                            validation_feedback.append(feedback)
                                except json.JSONDecodeError as e:
                                    validation_feedback.append(f"Failed to parse JSON response: {str(e)}")
                                
                                retry_count += 1
                            except Exception as e:
                                self.logger.error(f"Error in batch generation: {str(e)}")
                                retry_count += 1
                                continue
                        
                        if valid_batch_data:
                            all_synthetic_data.extend(valid_batch_data)
                            remaining_samples -= len(valid_batch_data)
                            print(f"  ✓ Batch {batch_num}: {len(valid_batch_data)} samples ({len(all_synthetic_data)}/{self.config.synthetic_data_size})")
                        else:
                            print(f"  ⚠️  Batch {batch_num} failed after {max_retries} retries")
                            break
                        
                        batch_num += 1
                    
                    self.llm_cost += sum([x['cost'] for x in getattr(tmp_lm, 'history', []) if x.get('cost') is not None])
                finally:
                    if 'tmp_lm' in locals():
                        del tmp_lm
                
            print(f"✅ Generated {len(all_synthetic_data)} synthetic samples")
            return all_synthetic_data

        except Exception as e:
            print(f"❌ Failed to generate synthetic data: {str(e)}")
            self.logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def _prepare_sample_data(self) -> Dict:
        """Prepare sample data for synthetic data generation."""
        if isinstance(self.config.sample_data, str):
            try:
                # First try to parse as JSON
                try:
                    data = json.loads(self.config.sample_data)
                    if isinstance(data, list):
                        return data[0], data
                    else:
                        return data, [data]
                except json.JSONDecodeError:
                    # If JSON parsing fails, try ast.literal_eval
                    data = ast.literal_eval(self.config.sample_data)
                    if isinstance(data, list):
                        return data[0], data
                    else:
                        return data, [data]
            except (SyntaxError, ValueError, json.JSONDecodeError) as e:
                self.logger.error(f"Error parsing sample data: {str(e)}")
                raise ValueError(f"Invalid sample data format: {str(e)}")
        elif isinstance(self.config.sample_data, list):
            return self.config.sample_data[0], self.config.sample_data
        elif isinstance(self.config.sample_data, dict):
            return self.config.sample_data, [self.config.sample_data]
        else:
            raise ValueError(f"Unexpected sample_data type: {type(self.config.sample_data)}")

    def _create_synthetic_data_prompt(self, sample_data: Dict, template: Dict, batch_size: int, feedback_section: str = "") -> str:
        """Generate a high-quality prompt for synthetic data creation with specified batch size."""
        
        return generate_synthetic_data_prompt(
            task=self.config.task,
            batch_size=batch_size,
            example_data=json.dumps(sample_data, indent=2),
            template=json.dumps([template], indent=2),
            feedback_section=feedback_section
        )

    def _clean_llm_response(self, response: str) -> str:
        """Clean and format LLM response."""
        if "```json" in response:
            response = response.split("```json")[1].strip()
        if "```" in response:
            response = response.split("```")[0].strip()
        return response.strip()

    def run(self, initial_flag: bool = True) -> Dict:
        """
        Run the optimization process using the configured backend.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        if self.backend == 'dspy':
            return self._run_dspy_backend(initial_flag)
        elif self.backend == 'simple_meta_prompt':
            return self._run_meta_prompt_backend(initial_flag)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _run_dspy_backend(self, initial_flag: bool = True) -> Dict:
        """
        Run optimization using DSPy backend.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        # Create a context to suppress I/O errors from logging during optimization
        import contextlib
        
        @contextlib.contextmanager
        def suppress_logging_errors():
            """Context manager to suppress I/O errors from logging."""
            # Store original error handlers
            original_callbacks = []
            for handler in logging.root.handlers + list(logging.Logger.manager.loggerDict.values()):
                if hasattr(handler, 'emit'):
                    original_emit = handler.emit
                    def safe_emit(record):
                        try:
                            original_emit(record)
                        except (IOError, OSError):
                            pass  # Silently ignore I/O errors
                    handler.emit = safe_emit
                    original_callbacks.append((handler, original_emit))
            try:
                yield
            finally:
                # Restore original handlers
                for handler, original_emit in original_callbacks:
                    try:
                        handler.emit = original_emit
                    except:
                        pass
        
        try:
            print("Starting DSPy optimization...")
            
            # Disable all file handlers for DSPy logger to prevent I/O conflicts
            import logging
            dspy_logger = logging.getLogger('dspy')
            dspy_handlers_backup = dspy_logger.handlers[:] if hasattr(dspy_logger, 'handlers') else []
            
            # Remove all file handlers from DSPy logger
            try:
                dspy_logger.handlers = [h for h in dspy_logger.handlers 
                                       if not isinstance(h, logging.FileHandler)]
            except:
                pass
            
            # Create signature
            print("Creating DSPy signature...")
            signature = self.create_signature(
                name=f"{self.config.task_type.upper()}Signature",
                input_fields=self.config.input_fields,
                output_fields=self.config.output_fields
            )
            print("✓ Signature created")

            # Generate synthetic data if needed
            if not self.config.train_data:
                print("Generating synthetic training data...")
                
                # Warn if dataset is too small for effective optimization
                min_recommended_size = 20  # Minimum recommended for MIPROv2
                if self.config.synthetic_data_size < min_recommended_size:
                    print(f"⚠ Warning: Dataset size ({self.config.synthetic_data_size}) is very small.")
                    print(f"⚠ MIPROv2 typically needs at least {min_recommended_size} samples for effective optimization.")
                    print(f"⚠ Consider using --synthetic_data_size {min_recommended_size} or higher for better results.")
                    print(f"⚠ Continuing with current size, but optimization may be limited...")
                
                synthetic_data = self.generate_synthetic_data()
                
                # Ensure train_data_size is calculated and at least 1
                if not hasattr(self.config, 'train_data_size') or self.config.train_data_size is None:
                    # Calculate dataset sizes if not already done
                    if hasattr(self.config, '_calculate_dataset_sizes'):
                        self.config._calculate_dataset_sizes()
                    else:
                        # Fallback: calculate manually
                        train_ratio = getattr(self.config, 'train_ratio', 0.2)
                        self.config.train_data_size = max(1, int(len(synthetic_data) * train_ratio))
                        self.config.valid_data_size = len(synthetic_data) - self.config.train_data_size
                
                # Ensure at least 1 sample in training set
                if self.config.train_data_size == 0 and len(synthetic_data) > 0:
                    self.config.train_data_size = 1
                    self.config.valid_data_size = len(synthetic_data) - 1
                
                self.config.train_data = synthetic_data[:self.config.train_data_size]
                self.config.valid_data = synthetic_data[self.config.train_data_size:]
                print(f"✓ Generated {len(synthetic_data)} synthetic samples")
                print(f"  Training samples: {len(self.config.train_data)}, Validation samples: {len(self.config.valid_data)}")

            # Prepare datasets
            print("Preparing datasets...")
            trainset, validset = self._prepare_datasets()
            validset_full = (self._prepare_full_validation_dataset() 
                           if self.config.valid_data_full else validset)
            print("✓ Datasets prepared")

            # Initialize trainer
            print("Initializing DSPy trainer...")
            trainer = self._initialize_trainer()
            print("✓ Trainer initialized")

            # Compile program
            print("Creating DSPy program...")
            if self.config.dspy_module == dspy.ReAct:
                program = self.config.dspy_module(signature, tools=self.config.tools)
            else:
                program = self.config.dspy_module(signature)
            print("✓ Program created")
            
            # Get evaluation metrics
            print("Setting up evaluation metrics...")
            eval_metrics = self.get_final_eval_metrics()
            print("✓ Evaluation metrics ready")
            
            # Evaluate initial prompt
            print("Evaluating initial prompt...")
            # Temporarily disable problematic file handlers during evaluation
            import logging
            dspy_logger = logging.getLogger('dspy')
            dspy_handlers_backup = dspy_logger.handlers[:] if hasattr(dspy_logger, 'handlers') else []
            
            # Remove file handlers from dspy logger to avoid I/O conflicts
            try:
                dspy_logger.handlers = [h for h in dspy_logger.handlers 
                                       if not isinstance(h, logging.FileHandler)]
            except:
                pass
            
            try:
                evaluator = Evaluate(devset=validset_full, metric=eval_metrics)
                initial_result = evaluator(program=program)
                initial_score = self._extract_score_from_result(initial_result)
                print(f"✓ Initial score: {initial_score:.4f}")
            except (IOError, OSError, ValueError) as err:
                # Handle I/O errors during evaluation (might be from logging)
                error_msg = str(err).lower()
                if 'closed' in error_msg or 'i/o operation' in error_msg:
                    # If it's a closed file error, try to continue with a default score
                    initial_score = 0.0
                    print(f"⚠ Warning: Evaluation encountered I/O error, using default score: {initial_score:.4f}")
                else:
                    raise  # Re-raise if it's a different error
            finally:
                # Restore dspy logger handlers
                try:
                    dspy_logger.handlers = dspy_handlers_backup
                except:
                    pass
            
            # Compile optimized program
            print("Compiling optimized program (this may take a while)...")
            # Temporarily disable ALL file handlers (including our own) during compilation
            # to prevent I/O conflicts
            import logging
            
            # Monkey-patch all file handlers to suppress I/O errors
            original_emit_methods = {}
            all_handlers = []
            
            # Collect all file handlers from all loggers
            for logger_name in logging.Logger.manager.loggerDict:
                logger_obj = logging.getLogger(logger_name)
                if hasattr(logger_obj, 'handlers'):
                    for handler in logger_obj.handlers:
                        if isinstance(handler, logging.FileHandler):
                            all_handlers.append(handler)
            
            # Also check root logger
            if hasattr(logging.root, 'handlers'):
                for handler in logging.root.handlers:
                    if isinstance(handler, logging.FileHandler):
                        all_handlers.append(handler)
            
            # Monkey-patch emit methods to suppress I/O errors
            def safe_emit_factory(original_emit):
                def safe_emit(record):
                    try:
                        original_emit(record)
                    except (IOError, OSError):
                        pass  # Silently ignore I/O errors
                return safe_emit
            
            for handler in all_handlers:
                if handler not in original_emit_methods:
                    original_emit_methods[handler] = handler.emit
                    handler.emit = safe_emit_factory(handler.emit)
            
            try:
                compiled_program = self._compile_program(trainer, program, trainset, validset)
                
                # Check if compilation actually produced an optimized program by comparing instructions
                try:
                    # Get instructions from compiled program
                    try:
                        compiled_instructions = compiled_program.signature.instructions
                    except:
                        try:
                            compiled_instructions = compiled_program.predict.signature.instructions
                        except:
                            compiled_instructions = None
                    
                    # Get instructions from original program
                    try:
                        original_instructions = program.signature.instructions
                    except:
                        try:
                            original_instructions = program.predict.signature.instructions
                        except:
                            original_instructions = None
                    
                    # Check if instructions changed (indicating optimization occurred)
                    if compiled_instructions and original_instructions:
                        if compiled_instructions == original_instructions:
                            print("⚠ Warning: Compilation completed but instructions unchanged (no optimization occurred)")
                            print("⚠ Possible reasons:")
                            print(f"  • Dataset too small ({len(trainset)} training, {len(validset)} validation samples)")
                            print(f"  • MIPROv2 couldn't find better instructions with current data")
                            print(f"  • Try increasing --synthetic_data_size to at least 20-30 for better results")
                            print("⚠ Note: Optimization may have improved few-shot examples even if instructions didn't change")
                        else:
                            print("✓ Program compilation complete - instructions optimized")
                            print(f"  Original: {original_instructions[:100]}...")
                            print(f"  Optimized: {compiled_instructions[:100]}...")
                    else:
                        # Can't compare instructions, assume compilation succeeded if we got a program
                        if compiled_program is not None:
                            print("✓ Program compilation complete")
                        else:
                            print("⚠ Warning: Compilation returned None")
                except Exception as check_err:
                    # If we can't check, assume compilation succeeded if we got a program
                    if compiled_program is not None:
                        print("✓ Program compilation complete")
                    else:
                        print("⚠ Warning: Could not verify compilation result")
            except (IOError, OSError) as io_err:
                # Handle I/O errors during compilation (likely from logging)
                error_msg = str(io_err).lower()
                if 'closed' in error_msg or 'i/o operation' in error_msg:
                    # Try to continue - compilation might have partially completed
                    print(f"⚠ Warning: I/O error during compilation (likely logging issue): {io_err}")
                    print("⚠ Attempting to use compiled program if available...")
                    # Try to get the program - _compile_program should have handled this
                    try:
                        compiled_program = self._compile_program(trainer, program, trainset, validset)
                    except:
                        compiled_program = program  # Final fallback
                    if compiled_program == program:
                        print("⚠ Using original program due to compilation error")
                else:
                    raise  # Re-raise if it's a different I/O error
            except Exception as e:
                # Catch any other errors during compilation
                error_msg = str(e).lower()
                if 'closed' in error_msg or 'i/o operation' in error_msg:
                    print(f"⚠ Warning: I/O error during compilation: {e}")
                    print("⚠ Attempting to use compiled program if available...")
                    try:
                        compiled_program = self._compile_program(trainer, program, trainset, validset)
                    except:
                        compiled_program = program
                    if compiled_program == program:
                        print("⚠ Using original program...")
                else:
                    raise  # Re-raise if it's not an I/O error
            finally:
                # Restore original emit methods
                for handler, original_emit in original_emit_methods.items():
                    try:
                        handler.emit = original_emit
                    except:
                        pass
            
            # Evaluate optimized prompt
            print("Evaluating optimized prompt...")
            # Temporarily disable problematic file handlers during evaluation
            dspy_logger = logging.getLogger('dspy')
            dspy_handlers_backup = dspy_logger.handlers[:] if hasattr(dspy_logger, 'handlers') else []
            
            # Remove file handlers from dspy logger to avoid I/O conflicts
            try:
                dspy_logger.handlers = [h for h in dspy_logger.handlers 
                                       if not isinstance(h, logging.FileHandler)]
            except:
                pass
            
            try:
                optimized_result = evaluator(program=compiled_program)
                optimized_score = self._extract_score_from_result(optimized_result)
                print(f"✓ Optimized score: {optimized_score:.4f}")
            except (IOError, OSError, ValueError) as err:
                # Handle I/O errors during evaluation (might be from logging)
                error_msg = str(err).lower()
                if 'closed' in error_msg or 'i/o operation' in error_msg:
                    # If it's a closed file error, try to continue with a default score
                    optimized_score = 0.0
                    print(f"⚠ Warning: Evaluation encountered I/O error, using default score: {optimized_score:.4f}")
                else:
                    raise  # Re-raise if it's a different error
            finally:
                # Restore dspy logger handlers
                try:
                    dspy_logger.handlers = dspy_handlers_backup
                except:
                    pass
            
            try:
                opt_instructions = compiled_program.signature.instructions
            except:
                opt_instructions = compiled_program.predict.signature.instructions
            
            # Prepare and return results
            print("Preparing final results...")
            # NOTE: DSPy can optimize demos/parameters without changing `signature.instructions`.
            # Track demo-count changes so users can see when optimization happened even if
            # "Original Prompt" and "Optimized Prompt" (instructions) look identical.
            try:
                original_demos_total = self._count_total_demos(program)
            except Exception:
                original_demos_total = 0
            try:
                optimized_demos_total = self._count_total_demos(compiled_program)
            except Exception:
                optimized_demos_total = 0

            result = self._prepare_results(
                self.config.task,
                opt_instructions,
                initial_score,
                optimized_score
            )
            result["optimization_debug"] = {
                "instructions_changed": False,
                "demos_changed": (original_demos_total != optimized_demos_total),
                "original_demos_total": original_demos_total,
                "optimized_demos_total": optimized_demos_total,
            }
            try:
                # Compare instructions best-effort
                original_instructions = program.signature.instructions
                result["optimization_debug"]["original_instructions"] = original_instructions
                result["optimization_debug"]["optimized_instructions"] = opt_instructions
                result["optimization_debug"]["instructions_changed"] = (original_instructions != opt_instructions)
            except Exception:
                pass
            
            # Calculate LLM cost from optimizer's LM if available
            if hasattr(self, 'lm') and self.lm is not None:
                self.llm_cost += sum([x['cost'] for x in getattr(self.lm, 'history', []) if x.get('cost') is not None])
            
            print("✓ DSPy optimization complete!")
            
            # Restore DSPy logger handlers before returning
            try:
                dspy_logger.handlers = dspy_handlers_backup
            except:
                pass
            
            return result

        except (IOError, OSError) as io_err:
            # Handle I/O errors (likely from logging) gracefully
            error_msg = str(io_err).lower()
            if 'closed' in error_msg or 'i/o operation' in error_msg:
                # This is likely a logging issue, try to continue with original program
                print(f"⚠ Warning: I/O error encountered (likely logging issue): {io_err}")
                print("⚠ Attempting to continue with original program...")
                try:
                    # Try to get the original program and return partial results
                    if 'program' in locals() and 'signature' in locals():
                        try:
                            opt_instructions = program.signature.instructions if hasattr(program, 'signature') else self.config.task
                        except:
                            opt_instructions = self.config.task
                        
                        result = self._prepare_results(
                            self.config.task,
                            opt_instructions,
                            0.0,  # initial_score
                            0.0   # optimized_score
                        )
                        
                        # Restore DSPy logger handlers before returning
                        try:
                            if 'dspy_logger' in locals() and 'dspy_handlers_backup' in locals():
                                dspy_logger.handlers = dspy_handlers_backup
                        except:
                            pass
                        
                        print("⚠ Returning partial results due to I/O error")
                        return result
                except Exception as e:
                    print(f"⚠ Could not recover from I/O error: {e}")
                
                # Return a minimal result
                return {
                    'error': f"I/O error (likely logging): {str(io_err)}",
                    'session_id': self.config.session_id,
                    'partial_result': True,
                    'optimized_prompt': self.config.task
                }
            else:
                # Re-raise if it's a different I/O error
                raise
        except Exception as e:
            # Catch any other exceptions that might be I/O errors in disguise
            error_msg = str(e).lower()
            if 'closed' in error_msg or 'i/o operation' in error_msg:
                print(f"⚠ Warning: I/O error encountered: {e}")
                print("⚠ Attempting to continue with original program...")
                # Try to return partial results
                try:
                    if 'program' in locals() and 'signature' in locals():
                        try:
                            opt_instructions = program.signature.instructions if hasattr(program, 'signature') else self.config.task
                        except:
                            opt_instructions = self.config.task
                        
                        result = self._prepare_results(
                            self.config.task,
                            opt_instructions,
                            0.0,
                            0.0
                        )
                        
                        # Restore DSPy logger handlers
                        try:
                            if 'dspy_logger' in locals() and 'dspy_handlers_backup' in locals():
                                dspy_logger.handlers = dspy_handlers_backup
                        except:
                            pass
                        
                        return result
                except:
                    pass
                
                return {
                    'error': f"I/O error: {str(e)}",
                    'session_id': self.config.session_id,
                    'partial_result': True,
                    'optimized_prompt': self.config.task
                }
            raise  # Re-raise if it's not an I/O error
        except Exception as e:
            print(f"❌ Error in DSPy optimization: {str(e)}")
            try:
                self.logger.error(f"Error in DSPy optimization run: {str(e)}")
            except:
                pass  # Don't let logger errors break the error reporting
            return {'error': str(e), 'session_id': self.config.session_id}

    def _run_meta_prompt_backend(self, initial_flag: bool = True) -> Dict:
        """
        Run optimization using meta-prompt backend with direct API calls.
        
        Args:
            initial_flag (bool): Whether this is the initial optimization
            
        Returns:
            Dict: Optimization results including metrics
        """
        try:
            # Generate synthetic data if needed (same as DSPy backend)
            if not self.config.train_data:
                synthetic_data = self.generate_synthetic_data()
                self.config.train_data = synthetic_data[:self.config.train_data_size]
                self.config.valid_data = synthetic_data[self.config.train_data_size:]
            
            # Evaluate initial prompt first
            print("🔧 Evaluating initial prompt...")
            initial_score = self._evaluate_prompt_meta_backend(self.config.task)
            print(f"  Initial score: {initial_score:.4f}")
            
            # Generate meta-prompt using the function from prompts.py
            meta_prompt = generate_meta_prompt_7(self.config.raw_input)
            # print("~"*100)
            # print(meta_prompt)
            # print("~"*100)
            
            # Get optimized prompt from LLM using direct API calls
            optimized_prompt = self._call_llm_api_directly(meta_prompt, "gpt-4.1")
            
            # Evaluate optimized prompt
            print("📊 Evaluating optimized prompt...")
            optimized_score = self._evaluate_prompt_meta_backend(optimized_prompt)
            print(f"  Optimized score: {optimized_score:.4f}")
            
            # Prepare and return results
            result = self._prepare_results(
                self.config.raw_input,
                optimized_prompt,
                initial_score,
                optimized_score
            )
            
            print("✅ Prompt optimization complete!")
            return result
                    
        except Exception as e:
            print(f"❌ Prompt optimization failed: {str(e)}")
            self.logger.error(f"Error in meta-prompt optimization run: {str(e)}")
            return {'error': str(e), 'session_id': self.config.session_id}

    def _evaluate_prompt_meta_backend(self, prompt: str) -> float:
        """
        Evaluate a prompt using the meta-prompt backend by testing it against synthetic data.
        
        Args:
            prompt (str): The prompt to evaluate
            
        Returns:
            float: Average score across all synthetic data samples
        """
        try:
            # Get the appropriate evaluation metric for the task type
            eval_metric = self.get_final_eval_metrics()
            
            # Use all available synthetic data for evaluation
            all_data = self.config.train_data + (self.config.valid_data or [])
            
            if not all_data:
                return 0.0
            
            total_score = 0.0
            valid_evaluations = 0
            
            for i, sample in enumerate(all_data):
                try:
                    # Create a test prompt by combining the prompt with the sample input
                    test_input = self._create_test_input_from_sample(sample)
                    full_test_prompt = f"{prompt}\n\n{test_input}"
                    
                    # Get prediction from LLM
                    prediction_text = self._call_llm_api_directly(full_test_prompt)
                    
                    # Create prediction object with the same structure as expected
                    prediction = self._create_prediction_object(prediction_text, sample)
                    
                    # Evaluate using the appropriate metric
                    score = eval_metric(sample, prediction, prompt)
                    total_score += score
                    valid_evaluations += 1
                    
                except Exception as e:
                    continue
            
            if valid_evaluations == 0:
                return 0.0
            
            average_score = total_score / valid_evaluations
            return average_score
            
        except Exception as e:
            self.logger.error(f"Error in prompt evaluation: {str(e)}")
            return 0.0

    def _create_test_input_from_sample(self, sample: Dict) -> str:
        """
        Create a test input string from a sample data dictionary.
        
        Args:
            sample (Dict): Sample data containing input fields
            
        Returns:
            str: Formatted test input string
        """
        try:
            # Parse input fields
            input_fields = self._parse_input_fields()
            
            if isinstance(input_fields, (list, tuple)):
                # Multiple input fields
                input_parts = []
                for field in input_fields:
                    if field in sample:
                        input_parts.append(f"{field}: {sample[field]}")
                    else:
                        input_parts.append(f"{field}: [MISSING]")
                return "\n".join(input_parts)
            else:
                # Single input field
                if input_fields in sample:
                    return f"{input_fields}: {sample[input_fields]}"
                else:
                    return f"{input_fields}: [MISSING]"
                
        except Exception as e:
            self.logger.error(f"Error creating test input: {str(e)}")
            # Fallback: return the sample as a simple string
            return str(sample)

    def _create_prediction_object(self, prediction_text: str, sample: Dict) -> Dict:
        """
        Create a prediction object with the expected structure for evaluation.
        
        Args:
            prediction_text (str): Raw prediction text from LLM
            sample (Dict): Original sample for reference
            
        Returns:
            Dict: Prediction object with output fields populated
        """
        try:
            # Parse output fields
            if isinstance(self.config.output_fields, str):
                output_fields = ast.literal_eval(self.config.output_fields)
            else:
                output_fields = self.config.output_fields
            
            # Create prediction object
            prediction = {}
            
            if isinstance(output_fields, (list, tuple)):
                # Multiple output fields - try to extract them from the prediction text
                if len(output_fields) == 1:
                    # Single output field, use the entire prediction text
                    prediction[output_fields[0]] = prediction_text.strip()
                else:
                    # Multiple output fields - this is more complex
                    # For now, assign the prediction text to the first field
                    # In a more sophisticated implementation, you might parse the text
                    prediction[output_fields[0]] = prediction_text.strip()
                    for field in output_fields[1:]:
                        prediction[field] = ""  # Empty for additional fields
            else:
                # Single output field
                prediction[output_fields] = prediction_text.strip()
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error creating prediction object: {str(e)}")
            # Fallback: return a simple object with the prediction text
            return {"output": prediction_text.strip()}

    def _call_llm_api_directly(self, prompt: str, model: str = "") -> str:
        """
        Call LLM API directly based on the configured provider.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The LLM response
        """
        try:
            # Determine the provider from config
            provider = getattr(self.config, 'config_model_provider', 'openai')
            if hasattr(provider, 'value'):
                provider = provider.value
            
            if provider.lower() == 'openai':
                oai_ouput = self._call_openai_api(prompt, model)
                return oai_ouput
            elif provider.lower() == 'anthropic':
                return self._call_anthropic_api(prompt)
            else:
                raise ValueError(f"Unsupported provider for direct API calls: {provider}")
                
        except Exception as e:
            self.logger.error(f"Error calling LLM API directly: {str(e)}")
            raise

    def _call_openai_api(self, prompt: str, model: str = "") -> str:
        """
        Call OpenAI API directly.
        
        Args:
            prompt (str): The prompt to send
            
        Returns:
            str: The API response
        """
        if model == "":
            model = self.config.config_model_name
        
        # Check if using Azure OpenAI
        azure_key = os.getenv("AZURE_OPENAI_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        use_azure = azure_key and azure_endpoint and deployment_name
        
        from openai import OpenAI
        
        # Configure OpenAI client
        if use_azure and (model.startswith("azure/") or not model):
            # Use Azure OpenAI
            azure_endpoint_clean = azure_endpoint.rstrip('/')
            # Azure OpenAI base URL format
            base_url = f"{azure_endpoint_clean}/openai/deployments/{deployment_name}"
            client = OpenAI(
                api_key=azure_key,
                base_url=base_url,
                default_query={"api-version": azure_api_version}
            )
            # Use deployment name as model name for Azure
            model = deployment_name
        else:
            # Use standard OpenAI
            client = OpenAI(
                api_key=self.config.config_model_api_key,
                base_url=self.config.config_model_api_base if self.config.config_model_api_base else None
            )


        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Calculate cost (approximate)
            # OpenAI pricing varies by model, this is a rough estimate
            input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
            output_tokens = len(response_text.split()) * 1.3
            
            # Estimate cost (this would need to be updated with actual pricing)
            estimated_cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)  # Rough estimate
            self.llm_cost += estimated_cost

            if model != "o3" and model != "gpt-4.1":
                return self._clean_llm_response(response_text)
            else:
                return response_text
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    def _call_anthropic_api(self, prompt: str) -> str:
        """
        Call Anthropic API directly.
        
        Args:
            prompt (str): The prompt to send
            
        Returns:
            str: The API response
        """
        import anthropic
        
        # Configure Anthropic client
        client = anthropic.Anthropic(
            api_key=self.config.config_model_api_key,
            base_url=self.config.config_model_api_base if self.config.config_model_api_base else None
        )
        
        try:
            response = client.messages.create(
                model=self.config.config_model_name,
                max_tokens=self.config.config_max_tokens,
                temperature=self.config.config_temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract response content
            response_text = response.content[0].text
            
            # Calculate cost (approximate)
            # Anthropic pricing varies by model, this is a rough estimate
            input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
            output_tokens = len(response_text.split()) * 1.3
            
            # Estimate cost (this would need to be updated with actual pricing)
            estimated_cost = (input_tokens * 0.000015) + (output_tokens * 0.000075)  # Rough estimate
            self.llm_cost += estimated_cost
            
            return self._clean_llm_response(response_text)
            
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            raise

    def _parse_input_fields(self) -> Union[str, List[str], Tuple[str, ...]]:
        """Parse input fields from config."""
        return (ast.literal_eval(self.config.input_fields) 
                if isinstance(self.config.input_fields, str) 
                else self.config.input_fields)

    def _prepare_dataset(self, data: List[Dict]) -> List[dspy.Example]:
        """Prepare a dataset from input data."""
        input_fields = self._parse_input_fields()
        
        if isinstance(input_fields, (list, tuple)):
            return [dspy.Example(**ex).with_inputs(*input_fields) for ex in data]
        return [dspy.Example(**ex).with_inputs(input_fields) for ex in data]

    def _prepare_datasets(self):
        """Prepare training and validation datasets."""
        return (
            self._prepare_dataset(self.config.train_data),
            self._prepare_dataset(self.config.valid_data)
        )

    def _prepare_full_validation_dataset(self):
        """Prepare full validation dataset if available."""
        return self._prepare_dataset(self.config.valid_data_full)

    def _extract_score_from_result(self, result) -> float:
        """
        Extract numeric score from DSPy evaluation result.
        
        Handles different return types from DSPy's Evaluate function:
        - Direct float/int values
        - EvaluationResult objects with score/average attributes
        - Other result types
        
        Args:
            result: Result from DSPy Evaluate function
            
        Returns:
            float: Extracted score value
        """
        if isinstance(result, (int, float)):
            return float(result)
        
        # Try common attributes
        if hasattr(result, 'score'):
            return float(result.score)
        if hasattr(result, 'average'):
            return float(result.average)
        
        # Try to convert directly
        try:
            return float(result)
        except (TypeError, ValueError):
            # Fallback: try to get any numeric attribute
            for attr in ['score', 'average', 'mean', 'value']:
                if hasattr(result, attr):
                    val = getattr(result, attr)
                    if isinstance(val, (int, float)):
                        return float(val)
            
            # Last resort: return 0.0
            self.logger.warning(f"Could not extract score from result type {type(result)}, returning 0.0")
            return 0.0

    def _count_total_demos(self, program) -> int:
        """Best-effort count of demos attached to a DSPy program/predictors."""
        if program is None:
            return 0

        total = 0
        predictors = []

        try:
            predictors = list(program.predictors())
        except Exception:
            # Fallback shapes
            for attr in ("predict", "predictor"):
                try:
                    maybe = getattr(program, attr, None)
                    if maybe is not None:
                        predictors = [maybe]
                        break
                except Exception:
                    continue

        for p in predictors or []:
            try:
                demos = getattr(p, "demos", None)
                if demos is not None:
                    total += len(demos)
            except Exception:
                continue

        # Some program variants keep demos at top-level
        try:
            demos = getattr(program, "demos", None)
            if demos is not None:
                total = max(total, len(demos))
        except Exception:
            pass

        return int(total)

    def _initialize_trainer(self):
        """Initialize the DSPy trainer."""
        return dspy.MIPROv2(
            metric=self.get_eval_metrics(),
            init_temperature=0.7,
            auto=self.config.miprov2_init_auto,
            num_candidates=self.config.miprov2_init_num_candidates
        )

    def _compile_program(self, trainer, program, trainset, validset):
        """Compile the program using the trainer."""
        # Suppress I/O errors more aggressively by wrapping the entire compile call
        import sys
        import contextlib
        
        # Create a context manager that suppresses I/O errors from stderr/stdout
        @contextlib.contextmanager
        def suppress_io_errors():
            """Suppress I/O errors that might occur during compilation."""
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            
            class SafeIO:
                """Wrapper that suppresses I/O errors."""
                def __init__(self, original):
                    self.original = original
                    self.buffer = getattr(original, 'buffer', None)
                
                def write(self, s):
                    try:
                        return self.original.write(s)
                    except (IOError, OSError):
                        pass  # Silently ignore I/O errors
                
                def flush(self):
                    try:
                        return self.original.flush()
                    except (IOError, OSError):
                        pass
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            try:
                # Only wrap if there are actual I/O errors
                sys.stderr = SafeIO(sys.stderr)
                sys.stdout = SafeIO(sys.stdout)
                yield
            finally:
                sys.stderr = original_stderr
                sys.stdout = original_stdout
        
        compiled = None
        compilation_completed = False
        compilation_error = None
        
        # Adjust MIPROv2 parameters for small datasets to prevent failures
        num_trials = self.config.miprov2_compile_num_trials
        max_bootstrapped = self.config.miprov2_compile_max_bootstrapped_demos
        max_labeled = self.config.miprov2_compile_max_labeled_demos
        
        # For very small datasets, reduce trials and demos to avoid overfitting
        total_samples = len(trainset) + len(validset)
        if total_samples < 10:
            # Very small dataset - reduce complexity
            num_trials = min(num_trials, 5)  # Reduce trials
            max_bootstrapped = min(max_bootstrapped, 2)  # Reduce bootstrapped demos
            max_labeled = min(max_labeled, 1)  # Reduce labeled demos
            print(f"⚠ Small dataset detected ({total_samples} samples). Adjusting MIPROv2 parameters:")
            print(f"  • num_trials: {num_trials}, max_bootstrapped_demos: {max_bootstrapped}, max_labeled_demos: {max_labeled}")
        elif total_samples < 20:
            # Small dataset - moderate reduction
            num_trials = min(num_trials, 8)
            max_bootstrapped = min(max_bootstrapped, 3)
            max_labeled = min(max_labeled, 2)
            print(f"⚠ Small dataset ({total_samples} samples). Using reduced MIPROv2 parameters for stability.")
        
        try:
            # Try compilation with I/O error suppression
            with suppress_io_errors():
                compiled = trainer.compile(
                    program,
                    trainset=trainset,
                    valset=validset,
                    requires_permission_to_run=False,
                    max_bootstrapped_demos=max_bootstrapped,
                    max_labeled_demos=max_labeled,
                    num_trials=num_trials,
                    minibatch_size=self.config.miprov2_compile_minibatch_size
                )
                compilation_completed = True
            
            # Check if compilation actually produced an optimized program
            # by comparing the instructions/signature, not just object identity
            if compiled is not None:
                try:
                    # Get instructions from compiled program
                    try:
                        compiled_instructions = compiled.signature.instructions
                    except:
                        try:
                            compiled_instructions = compiled.predict.signature.instructions
                        except:
                            compiled_instructions = None
                    
                    # Get instructions from original program
                    try:
                        original_instructions = program.signature.instructions
                    except:
                        try:
                            original_instructions = program.predict.signature.instructions
                        except:
                            original_instructions = None
                    
                    # If instructions are different, compilation succeeded
                    if compiled_instructions and original_instructions and compiled_instructions != original_instructions:
                        print(f"✓ Compilation succeeded: Instructions were optimized!")
                        return compiled
                    
                    # If we got a compiled program but instructions are same, check if demos changed
                    if compilation_completed:
                        # Check if demos changed (another form of optimization)
                        try:
                            original_demos = self._count_total_demos(program)
                            compiled_demos = self._count_total_demos(compiled)
                            if original_demos != compiled_demos:
                                print(f"✓ Compilation succeeded: Few-shot examples optimized ({original_demos} → {compiled_demos} demos)")
                                print(f"⚠ Instructions unchanged, but demos were optimized - this is still valid optimization!")
                            else:
                                print(f"⚠ Warning: Compilation completed but no optimization detected")
                                print(f"⚠ Instructions unchanged and demos unchanged")
                                print(f"⚠ With only {len(trainset)} training sample(s), MIPROv2 couldn't find better instructions")
                                print(f"⚠ This is expected - MIPROv2 needs at least 20-30 samples for effective optimization")
                        except:
                            # If we can't check demos, still return compiled program
                            pass
                        return compiled
                except Exception as check_err:
                    # If we can't check instructions, assume compilation succeeded if we got a result
                    if compilation_completed and compiled is not None:
                        print(f"⚠ Could not verify optimization (error: {check_err}), but compilation completed - using compiled program")
                        return compiled
            
            # If we get here, compilation might have failed
            if compiled is None:
                print("⚠ Warning: Compilation returned None")
                print(f"⚠ With only {len(trainset)} training sample(s), MIPROv2 may have failed")
                print(f"⚠ Consider using at least 20-30 samples for reliable optimization")
                return program
            else:
                print("⚠ Warning: Could not verify if compilation optimized the program")
                return compiled
                
        except (IOError, OSError) as io_err:
            # Handle I/O errors - check if compilation completed before the error
            error_msg = str(io_err).lower()
            if 'closed' in error_msg or 'i/o operation' in error_msg:
                # The error likely occurred after compilation succeeded (during logging)
                if compilation_completed and compiled is not None:
                    print(f"⚠ I/O error occurred after compilation (likely logging issue) - using compiled program")
                    # Even if instructions didn't change, the compiled program might have optimized demos
                    return compiled
                else:
                    # I/O error happened DURING compilation - this is more serious
                    print(f"⚠ I/O error occurred during compilation (likely logging issue)")
                    print(f"⚠ Compilation may have been interrupted. With only {len(trainset)} training sample(s),")
                    print(f"⚠ MIPROv2 cannot effectively optimize. Consider using at least 20-30 samples.")
                    # Try to return compiled if we have it, otherwise return original
                    if compiled is not None:
                        print(f"⚠ Returning partially compiled program (optimization may be incomplete)")
                        return compiled
                    return program
            else:
                raise  # Re-raise if it's a different I/O error
        except Exception as e:
            # Check if it's an I/O error wrapped in another exception
            error_msg = str(e).lower()
            if 'closed' in error_msg or 'i/o operation' in error_msg:
                # I/O error - check if compilation completed
                if compilation_completed and compiled is not None:
                    print(f"⚠ I/O error after compilation (likely logging issue) - using compiled program")
                    return compiled
                else:
                    print(f"⚠ I/O error during compilation (likely logging issue)")
                    print(f"⚠ With only {len(trainset)} training sample(s), optimization is severely limited.")
                    if compiled is not None:
                        return compiled
                    return program
            # For other exceptions, check if compilation completed
            if compilation_completed and compiled is not None:
                print(f"⚠ Error after compilation: {e} - using compiled program")
                return compiled
            # Otherwise, it's a real compilation error
            print(f"⚠ Error during compilation: {e}")
            # If it's a dataset-related error, provide helpful message
            if 'empty' in error_msg or 'insufficient' in error_msg or 'trainset' in error_msg:
                print(f"⚠ Dataset issue: {len(trainset)} training samples may be insufficient for MIPROv2")
            raise

    def _prepare_results(self, initial_prompt: str, optimized_prompt: str, 
                        initial_score: float, optimized_score: float) -> Dict:
        """Prepare the final results dictionary."""    
            
        return {    
            'result': optimized_prompt,
            'initial_prompt': initial_prompt,
            'session_id': self.config.session_id,
            'backend': self.backend,
            'metrics': {
                'initial_prompt_score': initial_score,
                'optimized_prompt_score': optimized_score
            },
            'synthetic_data': self.config.train_data + (self.config.valid_data or []),  # Include all synthetic data
            'llm_cost': self.llm_cost
        }

    def get_eval_metrics(self):
        """Get evaluation metrics for the task type."""
        if isinstance(self.config.output_fields, str):
            output_fields = ast.literal_eval(self.config.output_fields)
        else:
            output_fields = self.config.output_fields
        
        MetricsManager.configure(output_fields)
        return MetricsManager.get_metrics_for_task(self.config.task_type)
    
    def get_final_eval_metrics(self):
        """Get final evaluation metrics for the task type."""
        return MetricsManager.get_final_eval_metrics(self.config.task_type)

    
    def _validate_synthetic_data(self, data: Dict, task: str) -> Tuple[bool, str]:
        """
        Validate the generated synthetic data for quality and consistency.
        
        Args:
            data (Dict): The generated data sample to validate
            task (str): The task to validate the data against
            
        Returns:
            Tuple[bool, str]: (True if valid, feedback message)
        """        
        try:
            prompt = validate_synthetic_data(task, data, self.config.input_fields, self.config.output_fields)

            response = self._call_llm_api_directly(prompt)

            # read and clean the response which is expected to be in json format
            response = self._clean_llm_response(response)
            response = json.loads(response)

            try:
                is_valid = response['is_valid']
                feedback = response['feedback']
            except Exception as e:
                self.logger.error(f"Error parsing validation response: {str(e)}")
                return False, "Invalid validation response"

            if not is_valid:
                return False, feedback
            else:
                return True, "Sample passed all validation checks"
            
            
        except Exception as e:
            error_msg = f"Error in data validation: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

def setup_optimizer_logger():
    """Set up dedicated logger for optimization steps and results."""
    # Check if handler already exists to avoid duplicates
    if hasattr(setup_optimizer_logger, '_handler_added'):
        return
    
    logger.setLevel(logging.DEBUG)

    # Ensure the optimizer logs directory exists
    OPTIMIZER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create unique log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OPTIMIZER_LOGS_DIR / f"optimizer_{timestamp}.jsonl"

    # File Handler for JSON Lines format - completely fail-safe
    class JSONLinesHandler(logging.FileHandler):
        def __init__(self, filename, mode='a', encoding=None, delay=False):
            super().__init__(filename, mode, encoding, delay)
            self._lock = threading.Lock() if 'threading' in sys.modules else None
        
        def emit(self, record):
            # Completely wrap in try-except to prevent any errors from propagating
            try:
                # Use lock if available for thread safety
                if self._lock:
                    with self._lock:
                        self._emit_record(record)
                else:
                    self._emit_record(record)
            except:
                # Silently ignore ALL errors - don't let logging break the application
                pass
        
        def _emit_record(self, record):
            # Wrap everything in try-except to be completely fail-safe
            try:
                # Only try to parse JSON if it's an optimization step
                if hasattr(record, 'optimization_step'):
                    msg = record.optimization_step
                else:
                    # For regular log messages, create a simple JSON structure
                    msg = {
                        'timestamp': datetime.now().isoformat(),
                        'level': record.levelname,
                        'message': self.format(record)
                    }
                
                # Check if stream is available and open
                try:
                    if self.stream is None or (hasattr(self.stream, 'closed') and self.stream.closed):
                        self.stream = self._open()
                except:
                    return  # If we can't open, skip silently
                
                # Write to the file - catch ALL possible errors
                try:
                    json.dump(msg, self.stream)
                    self.stream.write('\n')
                    self.flush()
                except:
                    # If ANY error occurs (I/O, JSON, etc.), skip silently
                    return
            except:
                # Catch-all: silently ignore everything
                return

    # Custom formatter for optimization steps
    class OptimizationStepFormatter(logging.Formatter):
        def format(self, record):
            if hasattr(record, 'optimization_step'):
                return json.dumps(record.optimization_step)
            return super().format(record)

    # Set up file handler with custom formatter
    try:
        file_handler = JSONLinesHandler(log_file)
        file_handler.setFormatter(OptimizationStepFormatter())
        logger.addHandler(file_handler)
        setup_optimizer_logger._handler_added = True
        logger.info(f"Optimizer logger initialized. Log file: {log_file}")
    except:
        # If handler setup fails, continue without logging
        pass 