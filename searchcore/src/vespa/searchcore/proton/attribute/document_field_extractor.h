// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/stllike/hash_map.h>
#include <memory>
#include <string>

namespace document
{

class Document;
class FieldValue;
class FieldPath;
class FieldPathEntry;

}

namespace proton {

/**
 * Class used to extract a field value from a document field or from a
 * nested field in an array/map of structs.
 */
class DocumentFieldExtractor
{
    const document::Document &_doc;
    vespalib::hash_map<std::string, std::unique_ptr<document::FieldValue>> _cachedFieldValues;

    const document::FieldValue *getCachedFieldValue(const document::FieldPathEntry &fieldPathEntry);
    std::unique_ptr<document::FieldValue> getSimpleFieldValue(const document::FieldPath &fieldPath);
    std::unique_ptr<document::FieldValue> extractFieldFromStructArray(const document::FieldPath &fieldPath);
    std::unique_ptr<document::FieldValue> extractKeyFieldFromMap(const document::FieldPath &fieldPath);
    std::unique_ptr<document::FieldValue> extractValueFieldFromPrimitiveMap(const document::FieldPath &fieldPath);
    std::unique_ptr<document::FieldValue> extractValueFieldFromStructMap(const document::FieldPath &fieldPath);
<<<<<<< Updated upstream
    std::unique_ptr<document::FieldValue> extractSpecificValueFromMap(const document::FieldPath &fieldPath);
=======
    std::unique_ptr<document::FieldValue> extractSpecificKeyValueFromMap(const document::FieldPath &fieldPath);
>>>>>>> Stashed changes

public:
    DocumentFieldExtractor(const document::Document &doc);
    ~DocumentFieldExtractor();

    std::unique_ptr<document::FieldValue> getFieldValue(const document::FieldPath &fieldPath);

    /**
     * Check if fieldPath is in a supported form.
     */
    static bool isSupported(const document::FieldPath &fieldPath);
};

}
