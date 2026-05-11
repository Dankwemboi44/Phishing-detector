class PhishingDetectorError(Exception):
    """Base exception for phishing detector"""
    pass

class DataValidationError(PhishingDetectorError):
    """Raised when data validation fails"""
    pass

class ModelNotTrainedError(PhishingDetectorError):
    """Raised when trying to use untrained model"""
    pass

class PreprocessingError(PhishingDetectorError):
    """Raised during text preprocessing"""
    pass

class FeatureExtractionError(PhishingDetectorError):
    """Raised during feature extraction"""
    pass

class ConfigurationError(PhishingDetectorError):
    """Raised for configuration issues"""
    pass

def handle_exception(error: Exception, logger) -> dict:
    """Standard error response formatter"""
    logger.error(f"{error.__class__.__name__}: {str(error)}", exc_info=True)
    
    return {
        'error': error.__class__.__name__,
        'message': str(error),
        'success': False
    }