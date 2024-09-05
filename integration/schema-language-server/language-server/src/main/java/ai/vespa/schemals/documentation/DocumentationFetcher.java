package ai.vespa.schemals.documentation;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DocumentationFetcher
 */
public class DocumentationFetcher {
    private final static String SCHEMA_URL = "en/reference/schema-reference.html";
    private final static String RANK_FEATURE_URL = "en/reference/rank-features.html";

    private final static Map<String, List<String>> REPLACE_FILENAME_MAP = new HashMap<>(){{
        put("EXPRESSION", List.of( "EXPRESSION_SL", "EXPRESSION_ML" ));
        put("RANK_FEATURES", List.of( "RANKFEATURES_SL", "RANKFEATURES_ML" ));
        put("FUNCTION (INLINE)? [NAME]", List.of( "FUNCTION" ));
        put("SUMMARY_FEATURES", List.of( "SUMMARYFEATURES_SL", "SUMMARYFEATURES_ML", "SUMMARYFEATURES_ML_INHERITS" ));
        put("MATCH_FEATURES", List.of( "MATCHFEATURES_SL", "MATCHFEATURES_ML", "MATCHFEATURES_SL_INHERITS" ));
        put("IMPORT FIELD", List.of( "IMPORT" ));
    }};

    public static String fetchDocs() throws IOException {
        Path targetPath = Paths.get("").resolve("target").resolve("generated-resources").resolve("hover");
        Files.createDirectories(targetPath);
        Files.createDirectories(targetPath.resolve("schema"));
        Files.createDirectories(targetPath.resolve("rankExpression"));

        Path writePath = targetPath.resolve("schema");

        Map<String, String> schemaMarkdownContent = new SchemaDocumentationFetcher(SCHEMA_URL).getMarkdownContent();

        for (var entry : schemaMarkdownContent.entrySet()) {
            String fileName = convertToToken(entry.getKey());
            String content = entry.getValue();

            if (REPLACE_FILENAME_MAP.containsKey(fileName)) {
                for (String replacedFileName : REPLACE_FILENAME_MAP.get(fileName)) {
                    Files.write(writePath.resolve(replacedFileName + ".md"), content.getBytes(), StandardOpenOption.CREATE);
                }
            } else {
                Files.write(writePath.resolve(fileName + ".md"), content.getBytes(), StandardOpenOption.CREATE);
            }
        }

        Map<String, String> rankFeatureMarkdownContent = new RankFeatureDocumentationFetcher(RANK_FEATURE_URL).getMarkdownContent();

        writePath = targetPath.resolve("rankExpression");
        for (var entry : rankFeatureMarkdownContent.entrySet()) {
            Files.write(writePath.resolve(entry.getKey() + ".md"), entry.getValue().getBytes(), StandardOpenOption.CREATE);
        }

        return "LGTM";
    }

    private static String convertToToken(String h2Id) {
        return h2Id.toUpperCase().replaceAll("-", "_");
    }
}
