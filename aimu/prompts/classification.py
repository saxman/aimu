from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 3, "temperature": 0.01, "do_sample": True}

CLASSIFICATION_TASK_PROMPT_TEMPLATE = """
Read the following scientific paper abstract. Based on the content, determine if the paper explicitly refers to or uses a disease modeling technique,
including but not limited to mathematical, statistical, or computational methods used to simulate, analyze, predict, or interpret the dynamics of a disease.
"""

CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE = """
If the abstract describes or references any of these methods or similar approaches, answer "YES".
If the abstract focuses on non-modeling analysis, such as reporting observational data without reference to disease modeling techniques, answer "NO".
Wrap your final answer in square braces, e.g. [YES] or [NO]. Only include [YES] or [NO] in your answer.

Abstract:
{abstract}
"""


class ClassificationPromptTuner:
    def __init__(self, model_client, task_prompt: str = None):
        self.model_client = model_client
        self.initial_task_prompt = task_prompt
        self.tuned_prompt = None

        self.generate_kwargs = DEFAULT_GENERATE_KWARGS.copy()

        # Reasoning models need more generation tokens (for thinking) and a higher temperature for better results
        if model_client.model_id in model_client.THINKING_MODELS:
            self.generate_kwargs.update(
                {
                    "max_new_tokens": 1024,
                    "temperature": 0.6,
                }
            )

    def tune_prompt(self, training_data):
        """
        Executes the training process (prompt auto tuning) for classification using a hill climbing approach to optimize the prompt for 100% accuracy.

        Args:
            data: DataFrame with training data containing 'abstract' and 'is_modeling' columns

        Returns:
            Tuned prompt string
        """

        df_data = training_data.copy()
        df_data["predict_modeling"] = None

        # Use initial prompt if no tuned prompt exists
        if self.tuned_prompt is None:
            self.tuned_prompt = self.initial_task_prompt

        last_prompt = self.tuned_prompt
        iteration = mutation = 0

        # iterate using a basic hill climbing approach until a prompt is found that classifies w/100% accuracy
        while True:
            # Test current prompt
            df_data = self.classify_data(df_data)
            metrics = self.evaluate_results(df_data)

            out = f"results: iteration: {iteration}, mutation: {mutation} - precision: {metrics['precision']:.2f}. recall: {metrics['recall']:.2f}, accuracy: {metrics['accuracy']:.2f}"
            print(out)
            logger.info(out)

            iteration += 1

            # if accuracy has decreased, revert to the last prompt and try again
            if iteration > 1 and metrics["accuracy"] < self.last_metrics["accuracy"]:
                print("reverting to last prompt")
                self.tuned_prompt = last_prompt
                mutation -= 1
                continue

            df_bad = df_data.query("is_modeling != predict_modeling")

            # stop iterating once accuracy is 100%
            if len(df_bad) == 0:
                break

            # Store current state
            last_prompt = self.tuned_prompt
            self.last_metrics = metrics

            # randomly select an incorrect result for mutating the task prompt
            item = df_bad.sample().iloc[0]
            if item.is_modeling:
                mutation_prompt = self._get_positive_mutation_prompt(self.tuned_prompt, item.abstract)
            else:
                mutation_prompt = self._get_negative_mutation_prompt(self.tuned_prompt, item.abstract)

            print("mutating prompt")
            logger.info(f"mutation prompt:\n{mutation_prompt}")

            # generate a new prompt using the mutation prompt
            result = self.model_client.generate(mutation_prompt, self.generate_kwargs)
            logger.info(f"raw prompt mutation results:\n{result}")

            # extract the prompt from the response
            mutated_prompt = result.split("<prompt>")[-1].split("</prompt>")[0].strip()
            logger.info(f"processed prompt mutation results:\n{mutated_prompt}")

            mutation += 1
            self.tuned_prompt = mutated_prompt

        return self.tuned_prompt

    def classify_data(self, classification_prompt, data):
        """
        Execute classification task on the provided data using current prompt.

        Args:
            data: DataFrame with a 'content' column for classification.

        Returns:
            DataFrame with added 'predicted_class' column.
        """

        predicted_class = []
        df = data.copy()

        for i in tqdm(range(len(data)), desc="classifying"):
            row = data.iloc[i]
            prompt = classification_prompt.format(content=row.content)

            result = self.model_client.generate(prompt, self.generate_kwargs)

            logger.info(f"processed classification results: {result}")

            if "[yes]" in result.lower():
                predicted_class.append(True)
            elif "[no]" in result.lower():
                predicted_class.append(False)
            else:
                raise ValueError(f"unexpected classification result: {result}")

            logger.info(f"classification: {predicted_class[-1]}")

        df["predicted_class"] = predicted_class
        return df

    def evaluate_results(self, data):
        """
        Evaluate classification results and return metrics.

        Args:
            data: DataFrame with 'actual_class' and 'predicted_class' columns

        Returns:
            Dictionary with accuracy, precision, and recall metrics
        """

        true_pos = len(data.query("actual_class == True and predicted_class == True"))
        true_neg = len(data.query("actual_class == False and predicted_class == False"))

        accuracy = (true_pos + true_neg) / len(data)
        pos = len(data.query("predicted_class == True"))
        precision = true_pos / pos if pos > 0 else 0
        pos = len(data.query("actual_class == True"))
        recall = true_pos / pos if pos > 0 else 0

        return {"accuracy": accuracy, "precision": precision, "recall": recall}

    def _get_positive_mutation_prompt(self, prompt, abstract):
        """Generate mutation prompt for positive examples."""
        return f"""Given this task prompt:
{prompt}

This abstract should be classified as YES (modeling paper) but was classified as NO:
{abstract}

Please improve the task prompt to better identify modeling papers like this one. Return the improved prompt wrapped in <prompt></prompt> tags."""

    def _get_negative_mutation_prompt(self, prompt, abstract):
        """Generate mutation prompt for negative examples."""
        return f"""Given this task prompt:
{prompt}

This abstract should be classified as NO (non-modeling paper) but was classified as YES:
{abstract}

Please improve the task prompt to better exclude non-modeling papers like this one. Return the improved prompt wrapped in <prompt></prompt> tags."""
