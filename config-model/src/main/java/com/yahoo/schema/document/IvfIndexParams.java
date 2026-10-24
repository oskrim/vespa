// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.schema.document;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Parameters of the experimental disk-based IVF index, given as name/value pairs in an 'ivf' block of an index.
 * Names are passed to the content node unchanged (kebab case); the content node validates them.
 *
 * @author oskari
 */
public class IvfIndexParams {

    private final Map<String, String> params;

    public static class Builder {
        private final Map<String, String> params = new LinkedHashMap<>();

        public void set(String name, String value) { params.put(name, value); }

        public IvfIndexParams build() { return new IvfIndexParams(params); }
    }

    public IvfIndexParams(Map<String, String> params) {
        this.params = Collections.unmodifiableMap(new LinkedHashMap<>(params));
    }

    public Map<String, String> params() { return params; }

    @Override
    public boolean equals(Object o) {
        return (o instanceof IvfIndexParams other) && params.equals(other.params);
    }

    @Override
    public int hashCode() { return Objects.hash(params); }

    @Override
    public String toString() { return "ivf" + params; }

}
