#include <iostream>
#include <vespa/searchcore/proton/attribute/document_field_extractor.h>
#include <vespa/document/datatype/mapdatatype.h>
#include <vespa/document/datatype/primitive_data_type.h>
#include <vespa/document/fieldvalue/mapfieldvalue.h>
#include <vespa/document/fieldvalue/stringfieldvalue.h>
#include <vespa/document/fieldvalue/intfieldvalue.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/repo/documenttyperepo.h>

using namespace document;
using namespace proton;

int main() {
    std::cout << "Testing fieldpath sorting for map keys..." << std::endl;
    
    // Create a simple document type with a map field
    DocumentType docType("test", 42);
    auto keyType = DataType::STRING;
    auto valueType = DataType::INT;
    auto mapType = std::make_shared<MapDataType>(*keyType, *valueType);
    
    docType.addField(Field("myMap", *mapType));
    
    // Create a document with map data
    Document doc(docType, DocumentId("test::1"));
    MapFieldValue mapField(*mapType);
    
    // Add some key-value pairs
    StringFieldValue key1("key1");
    IntFieldValue value1(100);
    StringFieldValue key2("key2"); 
    IntFieldValue value2(200);
    
    mapField.put(key1, value1);
    mapField.put(key2, value2);
    doc.setValue("myMap", mapField);
    
    // Test field path parsing for specific key
    FieldPath fieldPath;
    try {
        docType.buildFieldPath(fieldPath, "myMap{key1}");
        std::cout << "Successfully parsed fieldpath: myMap{key1}" << std::endl;
        
        // Test if it's supported
        if (DocumentFieldExtractor::isSupported(fieldPath)) {
            std::cout << "✓ FieldPath for map key is supported" << std::endl;
            
            // Test extraction
            DocumentFieldExtractor extractor(doc);
            auto extractedValue = extractor.getFieldValue(fieldPath);
            if (extractedValue) {
                std::cout << "✓ Successfully extracted value from map key" << std::endl;
                if (extractedValue->isA(FieldValue::Type::INT)) {
                    auto intValue = static_cast<const IntFieldValue*>(extractedValue.get());
                    std::cout << "✓ Extracted value: " << intValue->getValue() << std::endl;
                }
            } else {
                std::cout << "✗ Failed to extract value from map key" << std::endl;
            }
        } else {
            std::cout << "✗ FieldPath for map key is not supported" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Failed to parse fieldpath: " << e.what() << std::endl;
    }
    
    return 0;
}