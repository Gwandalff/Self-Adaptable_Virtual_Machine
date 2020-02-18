package visitor.operation.miniJava;

import miniJava.visitor.miniJava.TypeRef;

public interface ClassRefOperation extends SingleTypeRefOperation {
	boolean compare(TypeRef other);
}
