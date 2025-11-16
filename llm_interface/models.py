from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class RadiologyReport(models.Model):
    """
    Stores original radiology report text from CSV uploads.
    Each report corresponds to one CSV entry.
    """
    entry_number = models.IntegerField(unique=True, db_index=True,
                                      help_text="Entry number from CSV file")
    report_text = models.TextField(help_text="Original radiology report text")
    csv_filename = models.CharField(max_length=255, null=True, blank=True,
                                   help_text="Source CSV filename")
    schema_name = models.CharField(max_length=100, default='radgraph',
                                  help_text="Schema used for this report")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['entry_number']
        verbose_name = "Radiology Report"
        verbose_name_plural = "Radiology Reports"

    def __str__(self):
        return f"Entry #{self.entry_number}"

    def get_ground_truth_triplets(self):
        """Get all ground truth triplets for this report"""
        return self.triplets.filter(source='ground_truth')

    def get_llm_triplets(self, function_id=None):
        """Get LLM-extracted triplets, optionally filtered by function"""
        if function_id:
            return self.triplets.filter(source=f'function_{function_id}')
        return self.triplets.filter(source__startswith='function_')


class ExtractionFunction(models.Model):
    """
    Stores custom Python code for different LLM extraction methods.
    Each function represents a different approach to extracting triplets.
    """
    name = models.CharField(max_length=255, unique=True,
                          help_text="Display name for this extraction function")
    description = models.TextField(blank=True,
                                  help_text="Description of what this function does")
    code = models.TextField(help_text="Python code for extraction function")
    is_active = models.BooleanField(default=True,
                                   help_text="Whether this function appears in tabs")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, null=True, blank=True,
                                  on_delete=models.SET_NULL,
                                  related_name='extraction_functions')

    # Display order for tabs
    order = models.IntegerField(default=0, help_text="Tab display order (lower = left)")

    class Meta:
        ordering = ['order', 'created_at']
        verbose_name = "Extraction Function"
        verbose_name_plural = "Extraction Functions"

    def __str__(self):
        return self.name

    def get_source_identifier(self):
        """Returns the source identifier used in Triplet.source field"""
        return f'function_{self.id}'


class Triplet(models.Model):
    """
    Stores entity-relation-entity triplets.
    Used for both ground truth (manual annotations) and LLM extractions.
    """
    SOURCE_GROUND_TRUTH = 'ground_truth'
    SOURCE_LEGACY_LLM = 'llm_legacy'  # For migrated entities.json data

    report = models.ForeignKey(RadiologyReport, on_delete=models.CASCADE,
                              related_name='triplets')
    source = models.CharField(max_length=100, db_index=True,
                            help_text="Source: 'ground_truth', 'function_<id>', or 'llm_legacy'")

    # Triplet fields
    entity1_text = models.CharField(max_length=500,
                                   help_text="First entity text span from report")
    entity1_type = models.CharField(max_length=100,
                                   help_text="Entity type from schema (e.g., 'Anatomy')")
    relation_type = models.CharField(max_length=100,
                                    help_text="Relation type from schema (e.g., 'located_at')")
    entity2_text = models.CharField(max_length=500,
                                   help_text="Second entity text span from report")
    entity2_type = models.CharField(max_length=100,
                                   help_text="Entity type from schema")

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, null=True, blank=True,
                                  on_delete=models.SET_NULL,
                                  related_name='created_triplets')

    # Optional fields
    confidence = models.FloatField(null=True, blank=True,
                                  help_text="LLM confidence or annotator confidence (0-1)")
    notes = models.TextField(blank=True, help_text="Additional notes or comments")

    class Meta:
        indexes = [
            models.Index(fields=['report', 'source']),
            models.Index(fields=['source']),
        ]
        verbose_name = "Triplet"
        verbose_name_plural = "Triplets"

    def __str__(self):
        return f"{self.entity1_text} -{self.relation_type}-> {self.entity2_text}"

    def to_dict(self):
        """Convert to dictionary format (compatible with entities.json format)"""
        return {
            'entity1_text': self.entity1_text,
            'entity1_type': self.entity1_type,
            'relation_type': self.relation_type,
            'entity2_text': self.entity2_text,
            'entity2_type': self.entity2_type,
            'confidence': self.confidence,
            'notes': self.notes,
        }

    @property
    def is_ground_truth(self):
        """Check if this is a ground truth triplet"""
        return self.source == self.SOURCE_GROUND_TRUTH

    @property
    def extraction_function(self):
        """Get the ExtractionFunction if this is from a function"""
        if self.source.startswith('function_'):
            function_id = self.source.replace('function_', '')
            try:
                return ExtractionFunction.objects.get(id=int(function_id))
            except (ExtractionFunction.DoesNotExist, ValueError):
                return None
        return None


class EvaluationRun(models.Model):
    """
    Stores results from an evaluation session comparing ground truth vs LLM extractions.
    Each run can evaluate one or more reports.
    """
    name = models.CharField(max_length=255, help_text="Name for this evaluation run")
    description = models.TextField(blank=True)

    # What was evaluated
    extraction_function = models.ForeignKey(ExtractionFunction, null=True, blank=True,
                                          on_delete=models.SET_NULL,
                                          related_name='evaluation_runs',
                                          help_text="Which extraction function was evaluated")
    schema_name = models.CharField(max_length=100, help_text="Schema used for evaluation")

    # Scope of evaluation
    entry_numbers = models.JSONField(help_text="List of entry numbers evaluated")
    total_reports = models.IntegerField(help_text="Number of reports evaluated")

    # Aggregate counts
    total_ground_truth_triplets = models.IntegerField(default=0)
    total_llm_triplets = models.IntegerField(default=0)
    exact_match_count = models.IntegerField(default=0)
    partial_match_count = models.IntegerField(default=0)

    # Aggregate metrics
    precision = models.FloatField(null=True, blank=True,
                                 help_text="Overall precision across all reports")
    recall = models.FloatField(null=True, blank=True,
                              help_text="Overall recall across all reports")
    f1_score = models.FloatField(null=True, blank=True,
                                help_text="Overall F1 score across all reports")

    # Detailed results (stored as JSON for flexibility)
    metrics_json = models.JSONField(null=True, blank=True,
                                   help_text="Detailed metrics breakdown (confusion matrices, per-type scores, etc.)")

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Evaluation Run"
        verbose_name_plural = "Evaluation Runs"

    def __str__(self):
        return f"{self.name} ({self.created_at.strftime('%Y-%m-%d')})"


class TripletComparison(models.Model):
    """
    Stores individual triplet comparison results.
    Links LLM-extracted triplets to their ground truth matches (if any).
    """
    MATCH_EXACT = 'exact'
    MATCH_ENTITY_ONLY = 'entity_only'
    MATCH_ENTITY_TYPE = 'entity_type'
    MATCH_PARTIAL = 'partial'
    MATCH_NONE = 'no_match'

    MATCH_TYPE_CHOICES = [
        (MATCH_EXACT, 'Exact Match - All 5 fields identical'),
        (MATCH_ENTITY_ONLY, 'Entity Match - Entity texts match only'),
        (MATCH_ENTITY_TYPE, 'Entity+Type Match - Entity texts and types match'),
        (MATCH_PARTIAL, 'Partial Match - Some similarity'),
        (MATCH_NONE, 'No Match'),
    ]

    evaluation_run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE,
                                      related_name='triplet_comparisons')

    # The triplets being compared
    llm_triplet = models.ForeignKey(Triplet, on_delete=models.CASCADE,
                                   related_name='llm_comparisons',
                                   help_text="LLM-extracted triplet")
    ground_truth_triplet = models.ForeignKey(Triplet, null=True, blank=True,
                                            on_delete=models.CASCADE,
                                            related_name='gt_comparisons',
                                            help_text="Matching ground truth triplet (if found)")

    # Match quality
    match_type = models.CharField(max_length=20, choices=MATCH_TYPE_CHOICES,
                                 default=MATCH_NONE)

    # Classification flags
    is_true_positive = models.BooleanField(default=False,
                                          help_text="LLM triplet matches ground truth")
    is_false_positive = models.BooleanField(default=False,
                                           help_text="LLM triplet has no ground truth match")
    is_false_negative = models.BooleanField(default=False,
                                           help_text="Ground truth triplet was missed by LLM")

    # Similarity scores (for advanced matching)
    entity1_similarity = models.FloatField(null=True, blank=True,
                                          help_text="Text similarity score for entity1")
    entity2_similarity = models.FloatField(null=True, blank=True,
                                          help_text="Text similarity score for entity2")
    overall_similarity = models.FloatField(null=True, blank=True,
                                          help_text="Overall similarity score")

    class Meta:
        indexes = [
            models.Index(fields=['evaluation_run', 'match_type']),
        ]
        verbose_name = "Triplet Comparison"
        verbose_name_plural = "Triplet Comparisons"

    def __str__(self):
        return f"{self.match_type}: {self.llm_triplet}"