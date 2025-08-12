package flock;

import java.io.IOException;
import java.nio.file.*;
import java.util.Properties;

public class Props {
    public static Properties load(String path) throws IOException {
        Properties p = new Properties();
        try (var in = Files.newInputStream(Path.of(path))) { p.load(in); }
        return p;
    }
    public static <T> T req(Properties p, String k, java.util.function.Function<String,T> f) {
        String v = p.getProperty(k);
        if (v == null) throw new IllegalArgumentException("Falta propiedad: " + k);
        return f.apply(v.trim());
    }
}
