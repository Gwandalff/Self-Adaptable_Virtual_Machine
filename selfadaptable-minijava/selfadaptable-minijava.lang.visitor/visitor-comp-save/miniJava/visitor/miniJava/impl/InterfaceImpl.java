package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.Interface;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class InterfaceImpl extends TypeDeclarationImpl implements Interface {
	protected InterfaceImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.INTERFACE;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Interface(this);
	}
}
