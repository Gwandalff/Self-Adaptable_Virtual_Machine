package visitor.operation.miniJava.impl;

import java.lang.Boolean;
import miniJava.visitor.miniJava.ClassRef;
import miniJava.visitor.miniJava.TypeRef;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ClassRefOperation;

public class ClassRefOperationImpl extends SingleTypeRefOperationImpl implements ClassRefOperation {
	private final ClassRef it;

	private final VisitorInterface vis;

	public ClassRefOperationImpl(ClassRef it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public boolean compare(TypeRef other) {
		boolean result;
		if(other instanceof ClassRef) {
			ClassRef ocr = ((ClassRef) (other));
			result = ((Boolean) (EqualService.equals((this.it.getReferencedClass()), (ocr.getReferencedClass()))));
		}
		else {
			result = ((Boolean) (false));
		}
		return result;
	}
}
