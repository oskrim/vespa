// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.schema.processing;

import com.yahoo.schema.derived.AttributeFields;
import com.yahoo.schema.document.Attribute;
import com.yahoo.schema.parser.ParseException;
import com.yahoo.vespa.config.search.AttributesConfig;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static com.yahoo.schema.ApplicationBuilder.createFromString;
import static com.yahoo.config.model.test.TestUtil.joinLines;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * @author geirst
 */
public class TensorFieldTestCase {

    @Test
    void requireThatTensorFieldCannotBeOfCollectionType() throws ParseException {
        try {
            createFromString(getSd("field f1 type array<tensor(x{})> {}"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("For schema 'test', field 'f1': A field with collection type of tensor is not supported. Use simple type 'tensor' instead.",
                    e.getMessage());
        }
    }

    @Test
    void requireThatTensorFieldCannotBeIndexField() throws ParseException {
        try {
            createFromString(getSd("field f1 type tensor(x{}) { indexing: index }"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("For schema 'test', field 'f1': A tensor of type 'tensor(x{})' does not support having an 'index'. " +
                    "Currently, only tensors with 1 indexed dimension or 1 mapped + 1 indexed dimension support that.",
                    e.getMessage());
        }
    }

    @Test
    void requireThatIndexedTensorAttributeCannotBeFastSearch() throws ParseException {
        try {
            createFromString(getSd("field f1 type tensor(x[3]) { indexing: attribute \n attribute: fast-search }"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("For schema 'test', field 'f1': An attribute of type 'tensor' cannot be 'fast-search'.", e.getMessage());
        }
    }

    @Test
    void requireThatIndexedTensorAttributeCannotBeFastRank() throws ParseException {
        try {
            createFromString(getSd("field f1 type tensor(x[3]) { indexing: attribute \n attribute: fast-rank }"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("attribute 'f1' (tensor(x[3])) does not support 'fast-rank'. Only supported for tensor types with at least one mapped dimension", e.getMessage());
        }
    }

    @Test
    void requireThatIllegalTensorTypeSpecThrowsException() throws ParseException {
        try {
            createFromString(getSd("field f1 type tensor(invalid) { indexing: attribute }"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertStartsWith("Field type: Illegal tensor type spec:", e.getMessage());
        }
    }

    @Test
    void hnsw_index_is_default_turned_off() throws ParseException {
        var attr = getAttributeFromSd("field t1 type tensor(x[64]) { indexing: attribute }", "t1");
        assertFalse(attr.hnswIndexParams().isPresent());
    }

    @Test
    void hnsw_index_gets_default_parameters_if_not_specified() throws ParseException {
        assertHnswIndexParams("", 16, 200);
        assertHnswIndexParams("index: hnsw", 16, 200);
    }

    @Test
    void tensor_with_one_mapped_and_one_indexed_dimension_can_have_hnsw_index() throws ParseException {
        assertHnswIndexParams("tensor(x{},y[64])", "", 16, 200);
        assertHnswIndexParams("tensor(x[64],y{})", "", 16, 200);
    }

    @Test
    void tensor_with_two_mapped_and_one_indexed_dimension_can_have_hnsw_index() throws ParseException {
        assertHnswIndexParams("tensor(x{},y{},z[64])", "", 16, 200);
        assertHnswIndexParams("tensor(x{},y[64],z{})", "", 16, 200);
        assertHnswIndexParams("tensor(x[64],y{},z{})", "", 16, 200);
    }

    @Test
    void hnsw_index_parameters_can_be_specified() throws ParseException {
        assertHnswIndexParams("index { hnsw { max-links-per-node: 32 } }", 32, 200);
        assertHnswIndexParams("index { hnsw { neighbors-to-explore-at-insert: 300 } }", 16, 300);
        assertHnswIndexParams(joinLines("index {",
                "  hnsw {",
                "    max-links-per-node: 32",
                "    neighbors-to-explore-at-insert: 300",
                "  }",
                "}"),
                32, 300);
    }

    @Test
    void tensor_with_hnsw_index_must_be_an_attribute() throws ParseException {
        try {
            createFromString(getSd("field t1 type tensor(x[64]) { indexing: index }"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("For schema 'test', field 't1': A tensor that has an index must also be an attribute.", e.getMessage());
        }
    }

    @Test
    void tensor_with_hnsw_index_parameters_must_be_an_index() throws ParseException {
        try {
            createFromString(getSd(joinLines(
                    "field t1 type tensor(x[64]) {",
                    "  indexing: attribute ",
                    "  index {",
                    "    hnsw { max-links-per-node: 32 }",
                    "  }",
                    "}")));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("For schema 'test', field 't1': " +
                    "A tensor that specifies hnsw index parameters must also specify 'index' in 'indexing'",
                    e.getMessage());
        }
    }

    @Test
    void ivf_index_is_default_turned_off() throws ParseException {
        var attr = getAttributeFromSd("field t1 type tensor(x[64]) { indexing: attribute | index }", "t1");
        assertFalse(attr.ivfIndexParams().isPresent());
        assertTrue(attr.hnswIndexParams().isPresent());
    }

    @Test
    void ivf_index_parameters_are_kept_as_given() throws ParseException {
        var attr = getAttributeFromSd(joinLines("field t1 type tensor<int8>(chunk{}, x[64]) {",
                "  indexing: attribute | index",
                "  attribute: paged",
                "  index {",
                "    ivf {",
                "      posting-capacity: 128",
                "      filter-probe-mode: adaptive",
                "      filter-exact-ratio: 0.05",
                "    }",
                "  }",
                "}"), "t1");
        assertTrue(attr.ivfIndexParams().isPresent());
        assertEquals(Map.of("posting-capacity", "128", "filter-probe-mode", "adaptive", "filter-exact-ratio", "0.05"),
                     attr.ivfIndexParams().get().params());
        assertEquals(List.of("posting-capacity", "filter-probe-mode", "filter-exact-ratio"),
                     List.copyOf(attr.ivfIndexParams().get().params().keySet()));
        // The field still has an index for everything that checks the hnsw parameters.
        assertTrue(attr.hnswIndexParams().isPresent());
    }

    @Test
    void ivf_index_without_parameters_uses_defaults() throws ParseException {
        var attr = getAttributeFromSd("field t1 type tensor(x[64]) { indexing: attribute | index \n index: ivf }", "t1");
        assertTrue(attr.ivfIndexParams().isPresent());
        assertTrue(attr.ivfIndexParams().get().params().isEmpty());
        attr = getAttributeFromSd("field t1 type tensor(x[64]) { indexing: attribute | index \n index { ivf { } } }", "t1");
        assertTrue(attr.ivfIndexParams().get().params().isEmpty());
    }

    @Test
    void ivf_index_is_written_to_attributes_config() throws ParseException {
        var schema = createFromString(getSd(joinLines("field t1 type tensor(x[64]) {",
                "  indexing: attribute | index",
                "  index { ivf { posting-capacity: 64 \n nprobe: 16 } }",
                "}",
                "field t2 type tensor(x[64]) { indexing: attribute | index }"))).getSchema();
        var builder = new AttributesConfig.Builder();
        new AttributeFields(schema).getConfig(builder);
        var config = builder.build();
        var t1 = config.attribute().stream().filter(a -> a.name().equals("t1")).findFirst().orElseThrow();
        assertTrue(t1.index().hnsw().enabled());
        assertTrue(t1.index().ivf().enabled());
        assertEquals(List.of("posting-capacity=64", "nprobe=16"),
                     t1.index().ivf().param().stream().map(p -> p.name() + "=" + p.value()).toList());
        var t2 = config.attribute().stream().filter(a -> a.name().equals("t2")).findFirst().orElseThrow();
        assertTrue(t2.index().hnsw().enabled());
        assertFalse(t2.index().ivf().enabled());
        assertTrue(t2.index().ivf().param().isEmpty());
    }

    @Test
    void tensor_with_ivf_index_parameters_must_be_an_index() throws ParseException {
        try {
            createFromString(getSd("field t1 type tensor(x[64]) { indexing: attribute \n index { ivf { nprobe: 8 } } }"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("For schema 'test', field 't1': A tensor that specifies an ivf index must also specify 'index' in 'indexing'",
                         e.getMessage());
        }
    }

    @Test
    void tensor_index_cannot_be_both_hnsw_and_ivf() throws ParseException {
        try {
            createFromString(getSd("field t1 type tensor(x[64]) { indexing: attribute | index \n index { hnsw \n ivf } }"));
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("For schema 'test', field 't1': A tensor index is either 'hnsw' or 'ivf', not both", e.getMessage());
        }
    }

    @Test
    void ivf_is_still_usable_as_a_name() throws ParseException {
        var attr = getAttributeFromSd("field ivf type tensor(x[4]) { indexing: attribute }", "ivf");
        assertFalse(attr.ivfIndexParams().isPresent());
    }

    @Test
    void tensors_with_at_least_one_mapped_dimension_can_be_direct() throws ParseException {
        assertTrue(getAttributeFromSd(
                "field t1 type tensor(x{}) { indexing: attribute \n attribute: fast-search }", "t1").isFastSearch());
        assertTrue(getAttributeFromSd(
                "field t1 type tensor(x{},y{},z[4]) { indexing: attribute \n attribute: fast-search }", "t1").isFastSearch());
    }

    @Test
    void tensors_with_at_least_one_mapped_dimension_can_be_fast_rank() throws ParseException {
        assertTrue(getAttributeFromSd(
                "field t1 type tensor(x{}) { indexing: attribute \n attribute: fast-rank }", "t1").isFastRank());
        assertTrue(getAttributeFromSd(
                "field t1 type tensor(x{},y{},z[4]) { indexing: attribute \n attribute: fast-rank }", "t1").isFastRank());
    }

    private static String getSd(String field) {
        return joinLines("search test {",
                "  document test {",
                "    " + field,
                "  }",
                "}");
    }

    private Attribute getAttributeFromSd(String fieldSpec, String attrName) throws ParseException {
        return createFromString(getSd(fieldSpec)).getSchema().getAttribute(attrName);
    }

    private void assertHnswIndexParams(String indexSpec, int maxLinksPerNode, int neighborsToExploreAtInsert) throws ParseException {
        assertHnswIndexParams("tensor(x[64])", indexSpec, maxLinksPerNode, neighborsToExploreAtInsert);
    }

    private void assertHnswIndexParams(String tensorType, String indexSpec, int maxLinksPerNode, int neighborsToExploreAtInsert) throws ParseException {
        var sd = getSdWithIndexSpec(tensorType, indexSpec);
        var search = createFromString(sd).getSchema();
        var attr = search.getAttribute("t1");
        var params = attr.hnswIndexParams();
        assertTrue(params.isPresent());
        assertEquals(maxLinksPerNode, params.get().maxLinksPerNode());
        assertEquals(neighborsToExploreAtInsert, params.get().neighborsToExploreAtInsert());
    }

    private String getSdWithIndexSpec(String tensorType, String indexSpec) {
        return getSd(joinLines("field t1 type " + tensorType + " {",
                "  indexing: attribute | index",
                "  " + indexSpec,
                "}"));
    }

    private void assertStartsWith(String prefix, String string) {
        assertEquals(prefix, string.substring(0, Math.min(prefix.length(), string.length())));
    }

}
