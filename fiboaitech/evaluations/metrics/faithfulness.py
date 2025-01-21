from typing import Any

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator

from fiboaitech.evaluations import BaseEvaluator
from fiboaitech.evaluations.llm_evaluator import LLMEvaluator
from fiboaitech.nodes.llms import BaseLLM
from fiboaitech.utils.logger import logger


class SimplifyStatementsInput(BaseModel):
    """
    Input model for simplifying statements.

    Attributes:
        questions (List[str]): List of questions.
        answers (List[str]): List of corresponding answers.
    """

    questions: list[str]
    answers: list[str]

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.questions) != len(self.answers):
            raise ValueError("Questions and answers must have the same length.")
        return self


class SimplifyStatementsOutput(BaseModel):
    """
    Output model for simplified statements.

    Attributes:
        statements_list (List[List[str]]): List of lists of simplified statements.
    """

    statements_list: list[list[str]]


class NLIInput(BaseModel):
    """
    Input model for NLI evaluation.

    Attributes:
        contexts (List[str]): List of contexts.
        statements_list (List[List[str]]): List of lists of statements.
    """

    contexts: list[str]
    statements_list: list[list[str]]

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.contexts) != len(self.statements_list):
            raise ValueError("Contexts and statements_list must have the same length.")
        return self


class NLIResultItem(BaseModel):
    """
    Model for individual NLI result.

    Attributes:
        statement (str): The statement being evaluated.
        verdict (int): 1 if faithful, 0 otherwise.
        reason (str): Reason for the verdict.
    """

    statement: str
    verdict: int
    reason: str

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v):
        if v not in (0, 1):
            raise ValueError("Verdict must be either 0 or 1.")
        return v


class NLIOutput(BaseModel):
    """
    Output model for NLI evaluation.

    Attributes:
        results_list (List[List[NLIResultItem]]): List of lists of NLI results.
    """

    results_list: list[list[NLIResultItem]]


class RunInput(BaseModel):
    """
    Input model for running the faithfulness evaluation.

    Attributes:
        questions (list[str]): List of questions.
        answers (list[str]): List of corresponding answers.
        contexts (list[str] | list[list[str]]): List of context texts for each question,
            which can be either one string per question (list[str]) or multiple
            strings per question (list[list[str]]) that will be joined.
        verbose (bool): Flag to enable verbose logging.
    """

    questions: list[str]
    answers: list[str]
    # Accept either list[str] or list[list[str]]
    contexts: list[str] | list[list[str]]
    verbose: bool = False

    @field_validator("contexts", mode="before")
    def unify_contexts(cls, value):
        """
        If contexts is list[list[str]], join each sublist with a space.
        Otherwise, if list[str], leave as-is.
        """
        if not isinstance(value, list):
            raise ValueError("contexts must be either a list of strings or a list of list of strings.")

        # Check if it's list[list[str]] => join each sublist
        if all(isinstance(item, list) and all(isinstance(x, str) for x in item) for item in value):
            return [" ".join(sublist) for sublist in value]

        # Check if it's already list[str]
        if all(isinstance(item, str) for item in value):
            return value

        raise ValueError("contexts must be either a list[str] or a list[list[str]].")

    @model_validator(mode="after")
    def check_equal_length(self):
        """
        By this time, self.contexts is guaranteed to be a list of strings.
        Ensure that questions, answers, and contexts all have the same length.
        """
        if not (len(self.questions) == len(self.answers) == len(self.contexts)):
            raise ValueError("Questions, answers, and contexts must have the same length.")
        return self


class RunOutput(BaseModel):
    """
    Output model for faithfulness evaluation.

    Attributes:
        final_scores (List[float]): List of faithfulness scores.
    """

    final_scores: list[float]


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluator class for faithfulness metric.

    Attributes:
        llm (BaseLLM): The language model to use for evaluation.
    """

    name: str = "Faithfulness"
    llm: BaseLLM

    _statement_simplifier: LLMEvaluator = PrivateAttr()
    _nli_evaluator: LLMEvaluator = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        simplify_instructions = (
            "Given a 'Question' and an 'Answer', break down each sentence in the "
            "Answer into one or more fully understandable statements.\n"
            "- Ensure no pronouns are used in each statement.\n"
            "- Output as a JSON object with key 'statements', where the value is a "
            "list of statements.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self._statement_simplifier = LLMEvaluator(
            instructions=simplify_instructions.strip(),
            inputs=[
                {"name": "question", "type": list[str]},
                {"name": "answer", "type": list[str]},
            ],
            outputs=[
                {"name": "statements", "type": list[str]},
            ],
            examples=[
                {
                    "inputs": {
                        "question": ["Who was Albert Einstein and what is he best known for?"],
                        "answer": [
                            (
                                "He was a German-born theoretical physicist, widely "
                                "acknowledged to be one of the greatest and most influential "
                                "physicists of all time. He was best known for developing "
                                "the theory of relativity, he also made important contributions "
                                "to the development of the theory of quantum mechanics."
                            )
                        ],
                    },
                    "outputs": {
                        "statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein is recognized as one of the greatest"
                            " and most influential physicists of all time.",
                            "Albert Einstein was best known for developing the theory of relativity.",
                            "Albert Einstein also made important contributions to the development"
                            " of quantum mechanics.",
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

        nli_instructions = (
            "Your task is to judge the faithfulness of a series of statements based "
            "on a given Context.\n"
            "- For each statement, return 'verdict': 1 if it can be directly inferred "
            "from the Context, or 0 if not.\n"
            "- Provide a brief 'reason' for the verdict.\n"
            "- Output as a JSON object with key 'results', where the value is a list "
            "of dictionaries with keys 'statement', 'verdict', and 'reason'.\n"
            "- Ensure your response is valid JSON, using double quotes for all strings."
        )

        self._nli_evaluator = LLMEvaluator(
            instructions=nli_instructions.strip(),
            inputs=[
                {"name": "context", "type": list[str]},
                {"name": "statements", "type": list[list[str]]},
            ],
            outputs=[
                {"name": "results", "type": list[dict[str, Any]]},
            ],
            examples=[
                {
                    "inputs": {
                        "context": [
                            (
                                "John is a student at XYZ University. He is pursuing a "
                                "degree in Computer Science. He is enrolled in several "
                                "courses this semester, including Data Structures, Algorithms, "
                                "and Database Management. John is a diligent student and "
                                "spends a significant amount of time studying and completing "
                                "assignments. He often stays late in the library to work on "
                                "his projects."
                            )
                        ],
                        "statements": [
                            [
                                "John is majoring in Biology.",
                                "John is taking a course on Artificial Intelligence.",
                                "John is a dedicated student.",
                                "John has a part-time job.",
                            ]
                        ],
                    },
                    "outputs": {
                        "results": [
                            {
                                "statement": "John is majoring in Biology.",
                                "verdict": 0,
                                "reason": "The context states that John is pursuing a degree"
                                " in Computer Science, not Biology.",
                            },
                            {
                                "statement": "John is taking a course on Artificial Intelligence.",
                                "verdict": 0,
                                "reason": "The context lists his courses,"
                                " and Artificial Intelligence is not mentioned.",
                            },
                            {
                                "statement": "John is a dedicated student.",
                                "verdict": 1,
                                "reason": "The context mentions he spends significant time"
                                " studying and stays late to work on projects.",
                            },
                            {
                                "statement": "John has a part-time job.",
                                "verdict": 0,
                                "reason": "There is no information in the context"
                                " about John having a part-time job.",
                            },
                        ]
                    },
                },
            ],
            llm=self.llm,
        )

    def simplify_statements(self, questions: list[str], answers: list[str]) -> list[list[str]]:
        """
        Simplify the answers into fully understandable statements.

        Args:
            questions (List[str]): List of questions.
            answers (List[str]): List of corresponding answers.

        Returns:
            List[List[str]]: List of lists of simplified statements.
        """
        input_data = SimplifyStatementsInput(questions=questions, answers=answers)
        results = self._statement_simplifier.run(
            question=input_data.questions,
            answer=input_data.answers,
        )
        statements_list = []
        for result in results["results"]:
            statements = result.get("statements")
            if isinstance(statements, list):
                statements_list.append(statements)
            else:
                statements_list.append([statements])
        output_data = SimplifyStatementsOutput(statements_list=statements_list)
        return output_data.statements_list

    def check_faithfulness(
        self,
        contexts: list[str],
        statements_list: list[list[str]],
    ) -> list[list[NLIResultItem]]:
        """
        Check the faithfulness of statements against contexts.

        Args:
            contexts (List[str]): List of contexts.
            statements_list (List[List[str]]): List of lists of statements.

        Returns:
            List[List[NLIResultItem]]: List of lists of NLI results.
        """
        input_data = NLIInput(contexts=contexts, statements_list=statements_list)
        results = self._nli_evaluator.run(
            context=input_data.contexts,
            statements=input_data.statements_list,
        )
        results_list = []
        for result in results["results"]:
            items = []
            for item in result["results"]:
                nli_item = NLIResultItem(
                    statement=item["statement"],
                    verdict=int(item["verdict"]),
                    reason=item["reason"],
                )
                items.append(nli_item)
            results_list.append(items)
        output_data = NLIOutput(results_list=results_list)
        return output_data.results_list

    def run(
        self,
        questions: list[str],
        answers: list[str],
        # Allow contexts to be either a list of strings or list of lists of strings
        contexts: list[str] | list[list[str]],
        verbose: bool = False,
    ) -> list[float]:
        """
        Evaluate the faithfulness of answers given contexts.

        Args:
            questions (list[str]): List of questions.
            answers (list[str]): List of corresponding answers.
            contexts (list[str] | list[list[str]]): List of context texts for each question,
                which can be passed either as list[str] (one context per question) or
                list[list[str]] (multiple context strings per question).
            verbose (bool): Flag to enable verbose logging.

        Returns:
            list[float]: List of faithfulness scores (one per question).
        """
        # Pass everything to RunInput
        input_data = RunInput(
            questions=questions,
            answers=answers,
            contexts=contexts,
            verbose=verbose,
        )

        final_scores = []

        for idx in range(len(input_data.questions)):
            question = input_data.questions[idx]
            answer = input_data.answers[idx]
            context = input_data.contexts[idx]  # Already guaranteed to be a single string

            # Simplify statements
            statements_list = self.simplify_statements([question], [answer])
            statements = statements_list[0]

            # Check faithfulness of statements
            results_list = self.check_faithfulness([context], [statements])
            results = results_list[0]

            # Compute faithfulness score
            num_statements = len(results)
            num_faithful = sum(item.verdict for item in results)
            score = num_faithful / num_statements if num_statements else 0.0
            final_scores.append(score)

            if input_data.verbose:
                logger.debug(f"Question: {question}")
                logger.debug(f"Answer: {answer}")
                logger.debug(f"Context: {context}")
                logger.debug("Simplified Statements:")
                logger.debug(statements)
                logger.debug("Faithfulness Results:")
                logger.debug([item.dict() for item in results])
                logger.debug(f"Faithfulness Score: {score}")
                logger.debug("-" * 50)

        output_data = RunOutput(final_scores=final_scores)
        return output_data.final_scores
