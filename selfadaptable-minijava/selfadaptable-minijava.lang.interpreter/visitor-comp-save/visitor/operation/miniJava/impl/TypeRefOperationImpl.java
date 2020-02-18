package visitor.operation.miniJava.impl;

import java.lang.Boolean;
import miniJava.visitor.miniJava.TypeRef;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.TypeRefOperation;

public class TypeRefOperationImpl implements TypeRefOperation {
	private final TypeRef it;

	private final VisitorInterface vis;

	public TypeRefOperationImpl(TypeRef it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}

	public boolean compare(TypeRef other) {
		boolean result;
		result = ((Boolean) (EqualService.equals((this.it.eClass()), (other.eClass()))));
		return result;
	}
}
