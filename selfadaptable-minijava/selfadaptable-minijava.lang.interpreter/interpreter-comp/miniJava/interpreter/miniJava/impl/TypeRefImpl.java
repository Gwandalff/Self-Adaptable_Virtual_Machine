package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.TypeRef;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class TypeRefImpl extends MinimalEObjectImpl.Container implements TypeRef {
	protected TypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.TYPE_REF;
	}

	public boolean compare(TypeRef other) {
		boolean result;
		result = (boolean) (EqualService.equals((this.eClass()), (other.eClass()))) ;
		return result;
	}
}
